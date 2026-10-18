import hashlib
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from lidar_codec.entropy import EntropyFrame, LaneState, LOWER, pack
from lidar_codec.sequence import SequenceReader, SequenceWriter


class ContainerContract(unittest.TestCase):
    def test_incremental_delivery_and_finish(self):
        prefix = b"LGC1" + bytes(324)
        body = b"payload"
        packet = prefix + struct.pack("<QI", len(body), zlib.crc32(body)) + body
        writer = SequenceWriter(11, prefix)
        chunks = (
            writer.header
            + writer.frame(0, True, packet)
            + writer.frame(1, True, packet)
            + writer.finish()
        )
        reader = SequenceReader(hashlib.sha256(prefix).hexdigest(), 11)
        frames = []
        for byte in chunks:
            frames.extend(reader.feed(bytes([byte])))
        self.assertEqual(reader.finish(), 2)
        self.assertEqual(frames, [(0, True, packet), (1, True, packet)])
        with self.assertRaises(ValueError):
            list(reader.feed(b"x"))

    def test_missing_end_and_identity_are_rejected(self):
        prefix = bytes(328)
        reader = SequenceReader(hashlib.sha256(prefix).hexdigest(), 12)
        with self.assertRaises(ValueError):
            list(reader.feed(SequenceWriter(11, prefix).header))
        reader = SequenceReader(hashlib.sha256(prefix).hexdigest(), 11)
        list(reader.feed(SequenceWriter(11, prefix).header))
        with self.assertRaises(ValueError):
            reader.finish()

    def test_entropy_crc_and_terminal_state_range(self):
        lanes = [LaneState(LOWER, b"") for _ in range(32)]
        stream = pack(lanes, active=True, carry=128, groups=6, symbols=50)
        self.assertEqual(EntropyFrame(stream, 128).expected, (6, 50))
        bad = bytearray(stream)
        bad[19] ^= 1
        with self.assertRaises(ValueError):
            EntropyFrame(bad, 128)
        lanes[0] = LaneState(0, b"")
        with self.assertRaises(ValueError):
            EntropyFrame(pack(lanes, active=True, carry=128, groups=6, symbols=50), 128)


class ManifestContract(unittest.TestCase):
    def test_full_split_and_missing_tail(self):
        path = Path(__file__).parents[1] / "tools/build_test_manifest.py"
        spec = importlib.util.spec_from_file_location("manifest_builder", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number in range(11, 22):
                sequence = root / f"{number:02d}"
                (sequence / "velodyne").mkdir(parents=True)
                (sequence / "times.txt").write_text("0\n0.1\n")
                for frame in range(2):
                    (sequence / "velodyne" / f"{frame:06d}.bin").write_bytes(bytes(16))
            self.assertEqual(len(module.build(root)["frames"]), 22)
            (root / "21/velodyne/000001.bin").unlink()
            with self.assertRaises(ValueError):
                module.build(root)


if __name__ == "__main__":
    unittest.main()
