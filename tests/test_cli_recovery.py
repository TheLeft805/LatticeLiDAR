"""CPU-only CLI ordering, bounded input and transactional output regressions."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zlib

import numpy as np

import cli_paths
import decode
import encode
from lidar_codec.io import write_points
from lidar_codec.sequence import SequenceReader, SequenceWriter


PREFIX = b"LGR1" + bytes(324)
POINTS = np.array([[1.25, -2.0, 0.5]], dtype=np.float32)


def packet(body=b"payload"):
    return PREFIX + struct.pack("<QI", len(body), zlib.crc32(body)) + body


def sequence_bytes(numbers=(1, 2), body=b"payload"):
    writer = SequenceWriter(0, PREFIX)
    return writer.header + b"".join(
        writer.frame(number, True, packet(body)) for number in numbers
    ) + writer.finish()


def fake_codec():
    codec = Mock(prefix=PREFIX)
    codec.encode.return_value = (packet(), {})
    codec.decode.return_value = (POINTS, {})
    return codec


class CliRecovery(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def inputs(self, names=("1.bin", "2.bin")):
        source = self.root / "inputs"
        source.mkdir(exist_ok=True)
        for name in names:
            (source / name).write_bytes(bytes(16))
        return source

    def invoke(self, module, source, output, codec=None, arguments=()):
        argv = [module.__name__, "--input", str(source), "--output", str(output), *arguments]
        self.factory = Mock(return_value=codec or fake_codec())
        stdout = io.StringIO()
        with (
            patch.object(sys, "argv", argv),
            patch.object(module, "create_codec", self.factory),
            redirect_stdout(stdout),
            redirect_stderr(io.StringIO()),
        ):
            module.main()
        return json.loads(stdout.getvalue())

    def assert_no_staging(self, output, preserved=()):
        self.assertEqual(
            set(output.parent.glob(f".{output.name}.partial-*")), set(preserved)
        )

    def preserve_unrelated_temporary(self, output):
        legacy = output.with_name(output.name + ".partial")
        legacy.write_bytes(b"earlier user data")
        unrelated = output.with_name(f".{output.name}.partial-foreign")
        unrelated.mkdir()
        (unrelated / "keep.txt").write_text("do not remove")
        return legacy, unrelated

    def assert_unrelated_preserved(self, output, legacy, unrelated):
        self.assertEqual(legacy.read_bytes(), b"earlier user data")
        self.assertEqual((unrelated / "keep.txt").read_text(), "do not remove")
        self.assert_no_staging(output, (unrelated,))

    def test_encode_sorts_numeric_ids_and_preserves_gaps(self):
        source = self.inputs(("10.bin", "2.bin", "1.bin"))
        output = self.root / "sequence.lsq"
        result = self.invoke(encode, source, output)
        self.assertEqual([row["input"] for row in result["frames"]], ["1.bin", "2.bin", "10.bin"])
        reader = SequenceReader(hashlib.sha256(PREFIX).hexdigest(), 0)
        self.assertEqual([frame[0] for frame in reader.feed(output.read_bytes())], [1, 2, 10])
        self.assertEqual(reader.finish(), 3)
        self.assert_no_staging(output)

    def test_encode_accepts_u32_boundaries(self):
        source = self.inputs(("4294967295.bin", "000000.bin"))
        output = self.root / "sequence.lsq"
        self.invoke(encode, source, output)
        reader = SequenceReader(hashlib.sha256(PREFIX).hexdigest(), 0)
        self.assertEqual([frame[0] for frame in reader.feed(output.read_bytes())], [0, 0xFFFFFFFF])
        self.assertEqual(reader.finish(), 2)

    def test_duplicate_numeric_ids_fail_before_gpu_initialization(self):
        source = self.inputs(("1.bin", "000001.bin"))
        output = self.root / "sequence.lsq"
        with self.assertRaisesRegex(ValueError, "Duplicate frame ID"):
            self.invoke(encode, source, output)
        self.factory.assert_not_called()
        self.assertFalse(output.exists())
        self.assert_no_staging(output)

    def test_bad_names_and_out_of_range_ids_fail_before_gpu_initialization(self):
        source = self.inputs(())
        output = self.root / "sequence.lsq"
        for name in ("frame.bin", "-1.bin", "+1.bin", "1.5.bin", "１.bin", "4294967296.bin", "999999999999999.bin"):
            with self.subTest(name=name):
                bad = source / name
                bad.touch()
                try:
                    with self.assertRaises(ValueError):
                        self.invoke(encode, source, output)
                    self.factory.assert_not_called()
                    self.assertFalse(output.exists())
                    self.assert_no_staging(output)
                finally:
                    bad.unlink()
        (source / "2.bin").mkdir()
        with self.assertRaises(ValueError):
            self.invoke(encode, source, output)
        self.factory.assert_not_called()

    def test_empty_sequence_fails_before_gpu_initialization(self):
        with self.assertRaisesRegex(ValueError, "No input frames"):
            self.invoke(encode, self.inputs(()), self.root / "sequence.lsq")
        self.factory.assert_not_called()

    def test_invalid_sequence_ids_fail_before_gpu_initialization(self):
        source = self.inputs()
        for module in (encode, decode):
            for value in ("-1", "4294967296", "not-a-number"):
                with self.subTest(module=module.__name__, value=value):
                    with self.assertRaises(SystemExit):
                        self.invoke(module, source, self.root / "out", arguments=("--sequence", value))
                    self.factory.assert_not_called()

    def test_encode_second_frame_failure_cleans_only_own_staging_and_retries(self):
        source = self.inputs()
        output = self.root / "sequence.lsq"
        legacy, unrelated = self.preserve_unrelated_temporary(output)
        codec = fake_codec()
        codec.encode.side_effect = [(packet(), {}), RuntimeError("second frame")]
        with self.assertRaisesRegex(RuntimeError, "second frame"):
            self.invoke(encode, source, output, codec)
        codec.close.assert_called_once()
        self.assertFalse(output.exists())
        self.assert_unrelated_preserved(output, legacy, unrelated)
        self.invoke(encode, source, output)
        self.assertTrue(output.is_file())
        self.assert_unrelated_preserved(output, legacy, unrelated)

    def test_decode_second_frame_failure_cleans_only_own_staging_and_retries(self):
        source = self.root / "sequence.lsq"
        source.write_bytes(sequence_bytes())
        output = self.root / "decoded"
        legacy, unrelated = self.preserve_unrelated_temporary(output)
        codec = fake_codec()
        codec.decode.side_effect = [(POINTS, {}), RuntimeError("second frame")]
        with self.assertRaisesRegex(RuntimeError, "second frame"):
            self.invoke(decode, source, output, codec)
        codec.close.assert_called_once()
        self.assertFalse(output.exists())
        self.assert_unrelated_preserved(output, legacy, unrelated)
        result = self.invoke(decode, source, output)
        self.assertEqual(result, dict(frames=2, complete=True))
        self.assertEqual(sorted(path.name for path in output.iterdir()), ["000001.ply", "000002.ply"])
        self.assert_unrelated_preserved(output, legacy, unrelated)

    def test_decode_missing_end_does_not_publish_earlier_frames_and_retries(self):
        source = self.root / "sequence.lsq"
        complete = sequence_bytes()
        source.write_bytes(complete[:-20])
        output = self.root / "decoded"
        codec = fake_codec()
        with self.assertRaisesRegex(ValueError, "Truncated"):
            self.invoke(decode, source, output, codec)
        self.assertEqual(codec.decode.call_count, 2)
        codec.close.assert_called_once()
        self.assertFalse(output.exists())
        self.assert_no_staging(output)
        source.write_bytes(complete)
        self.invoke(decode, source, output)
        self.assertEqual(len(list(output.iterdir())), 2)

    def test_existing_outputs_including_dangling_symlinks_are_untouched(self):
        source = self.root / "input.bin"
        source.write_bytes(bytes(16))
        for module in (encode, decode):
            for kind in ("file", "directory", "symlink"):
                with self.subTest(module=module.__name__, kind=kind):
                    output = self.root / f"{module.__name__}-{kind}.npy"
                    if kind == "file":
                        output.write_bytes(b"user output")
                    elif kind == "directory":
                        output.mkdir()
                        (output / "keep").write_bytes(b"user output")
                    else:
                        output.symlink_to(self.root / "missing-target")
                    inode = output.lstat().st_ino
                    with self.assertRaises(FileExistsError):
                        self.invoke(module, source, output)
                    self.factory.assert_not_called()
                    self.assertEqual(output.lstat().st_ino, inode)
                    if kind == "file":
                        self.assertEqual(output.read_bytes(), b"user output")
                    elif kind == "directory":
                        self.assertEqual((output / "keep").read_bytes(), b"user output")
                    else:
                        self.assertEqual(os.readlink(output), str(self.root / "missing-target"))
                    self.assert_no_staging(output)

    def test_file_publication_never_overwrites_racing_output(self):
        source = self.root / "input.bin"
        source.write_bytes(bytes(16))
        for module in (encode, decode):
            with self.subTest(module=module.__name__):
                output = self.root / f"{module.__name__}.npy"
                codec = fake_codec()

                def produce(_):
                    self.assertFalse(output.exists())
                    output.write_bytes(b"concurrent output")
                    return (packet(), {}) if module is encode else (POINTS, {})

                getattr(codec, module.__name__).side_effect = produce
                with self.assertRaises(FileExistsError):
                    self.invoke(module, source, output, codec)
                self.assertEqual(output.read_bytes(), b"concurrent output")
                codec.close.assert_called_once()
                self.assert_no_staging(output)

    def test_directory_publication_preserves_racing_empty_and_nonempty_outputs(self):
        source = self.root / "sequence.lsq"
        source.write_bytes(sequence_bytes((1,)))
        for kind in ("empty", "nonempty", "symlink", "file"):
            with self.subTest(kind=kind):
                output = self.root / kind
                codec = fake_codec()
                inode = []

                def produce(_):
                    self.assertFalse(os.path.lexists(output))
                    if kind == "symlink":
                        output.symlink_to(self.root / "missing-target")
                    elif kind == "file":
                        output.write_bytes(b"concurrent output")
                    else:
                        output.mkdir()
                        if kind == "nonempty":
                            (output / "keep").write_bytes(b"concurrent output")
                    inode.append(output.lstat().st_ino)
                    return POINTS, {}

                codec.decode.side_effect = produce
                with self.assertRaises(FileExistsError):
                    self.invoke(decode, source, output, codec)
                self.assertEqual(output.lstat().st_ino, inode[0])
                if kind == "empty":
                    self.assertEqual(list(output.iterdir()), [])
                elif kind == "nonempty":
                    self.assertEqual((output / "keep").read_bytes(), b"concurrent output")
                elif kind == "file":
                    self.assertEqual(output.read_bytes(), b"concurrent output")
                else:
                    self.assertEqual(os.readlink(output), str(self.root / "missing-target"))
                codec.close.assert_called_once()
                self.assert_no_staging(output)

    def test_single_decode_partial_write_failure_is_removed_and_retry_succeeds(self):
        source = self.root / "frame.lgr"
        source.write_bytes(packet())
        output = self.root / "frame.npy"
        codec = fake_codec()

        def fail_write(path, _):
            Path(path).write_bytes(b"partial numpy file")
            raise OSError("disk full")

        with patch.object(decode, "write_points", side_effect=fail_write):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.invoke(decode, source, output, codec)
        codec.close.assert_called_once()
        self.assertFalse(output.exists())
        self.assert_no_staging(output)
        self.invoke(decode, source, output)
        np.testing.assert_array_equal(np.load(output), POINTS)

    def test_sequence_second_write_failure_is_removed_and_retry_succeeds(self):
        source = self.root / "sequence.lsq"
        source.write_bytes(sequence_bytes())
        output = self.root / "decoded"
        calls = []

        def fail_second_write(path, points):
            calls.append(path)
            write_points(path, points)
            if len(calls) == 2:
                raise OSError("second write")

        with patch.object(decode, "write_points", side_effect=fail_second_write):
            with self.assertRaisesRegex(OSError, "second write"):
                self.invoke(decode, source, output)
        self.assertFalse(output.exists())
        self.assert_no_staging(output)
        self.invoke(decode, source, output)
        self.assertEqual(len(list(output.iterdir())), 2)

    def test_keyboard_interrupt_and_close_failure_do_not_publish(self):
        source = self.root / "input.bin"
        source.write_bytes(bytes(16))
        for module in (encode, decode):
            for failure in ("interrupt", "close"):
                with self.subTest(module=module.__name__, failure=failure):
                    output = self.root / f"{module.__name__}-{failure}.npy"
                    codec = fake_codec()
                    if failure == "interrupt":
                        getattr(codec, module.__name__).side_effect = KeyboardInterrupt
                        error = KeyboardInterrupt
                    else:
                        codec.close.side_effect = RuntimeError("close failure")
                        error = RuntimeError
                    with self.assertRaises(error):
                        self.invoke(module, source, output, codec)
                    codec.close.assert_called_once()
                    self.assertFalse(output.exists())
                    self.assert_no_staging(output)

    def test_codec_initialization_failure_cleans_staging_and_allows_retry(self):
        source = self.root / "input.bin"
        source.write_bytes(bytes(16))
        for module in (encode, decode):
            with self.subTest(module=module.__name__):
                output = self.root / f"{module.__name__}.npy"
                argv = [module.__name__, "--input", str(source), "--output", str(output)]
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(module, "create_codec", side_effect=RuntimeError("initialization")),
                ):
                    with self.assertRaisesRegex(RuntimeError, "initialization"):
                        module.main()
                self.assertFalse(output.exists())
                self.assert_no_staging(output)
                self.invoke(module, source, output)
                self.assertTrue(output.is_file())

    def test_invalid_single_decode_extension_fails_before_gpu_initialization(self):
        source = self.root / "frame.lgr"
        source.write_bytes(packet())
        with self.assertRaisesRegex(ValueError, "Reconstruction must"):
            self.invoke(decode, source, self.root / "out.xyz")
        self.factory.assert_not_called()

    def test_missing_atomic_directory_primitive_fails_without_publication(self):
        source = self.root / "sequence.lsq"
        source.write_bytes(sequence_bytes())
        output = self.root / "decoded"
        with patch.object(cli_paths.ctypes, "CDLL", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(OSError, "renameat2"):
                self.invoke(decode, source, output)
        self.assertFalse(output.exists())
        self.assert_no_staging(output)
        self.invoke(decode, source, output)
        self.assertEqual(len(list(output.iterdir())), 2)

    def test_single_frame_body_limit_excludes_wrapper_and_is_checked_before_gpu(self):
        source = self.root / "frame.lgr"
        output = self.root / "frame.npy"
        source.write_bytes(packet(b"ab"))
        with self.assertRaisesRegex(ValueError, "max-frame-bytes"):
            self.invoke(decode, source, output, arguments=("--max-frame-bytes", "1"))
        self.factory.assert_not_called()
        self.assertFalse(output.exists())
        self.assert_no_staging(output)
        self.invoke(decode, source, output, arguments=("--max-frame-bytes", "2"))
        np.testing.assert_array_equal(np.load(output), POINTS)

    def test_single_frame_input_read_is_bounded(self):
        source = self.root / "frame.lgr"
        output = self.root / "frame.npy"
        reads = []

        class TrackedInput(io.BytesIO):
            def read(self, size=-1):
                reads.append(size)
                return super().read(size)

        original_open = Path.open

        def open_path(path, *args, **kwargs):
            if path == source:
                return TrackedInput(packet(b"ab"))
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", open_path):
            self.invoke(decode, source, output, arguments=("--max-frame-bytes", "2"))
        self.assertEqual(reads, [4, 343])

    def test_sequence_body_limit_matches_single_frame_limit(self):
        source = self.root / "sequence.lsq"
        source.write_bytes(sequence_bytes(body=b"ab"))
        output = self.root / "decoded"
        codec = fake_codec()
        with self.assertRaisesRegex(ValueError, "frame extent"):
            self.invoke(decode, source, output, codec, arguments=("--max-frame-bytes", "1"))
        codec.decode.assert_not_called()
        codec.close.assert_called_once()
        self.assertFalse(output.exists())
        self.assert_no_staging(output)
        self.invoke(decode, source, output, arguments=("--max-frame-bytes", "2"))
        self.assertEqual(len(list(output.iterdir())), 2)

    def test_invalid_frame_limits_fail_before_gpu_initialization(self):
        for limit in ("0", "-1", "bad", str(sys.maxsize)):
            with self.subTest(limit=limit):
                with self.assertRaises(SystemExit):
                    self.invoke(decode, self.root / "absent", self.root / "out.npy", arguments=("--max-frame-bytes", limit))
                self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
