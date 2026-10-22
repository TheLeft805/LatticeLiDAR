"""CPU-only contracts for portable dataset identity and global sampling."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


BUILDER_PATH = Path(__file__).parents[1] / "tools/build_test_manifest.py"
SPEC = importlib.util.spec_from_file_location("manifest_identity_builder", BUILDER_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def make_dataset(root, reverse=False):
    """Small, complete split with enough frames to cross a global stride."""
    sequences = list(range(11, 22))
    if reverse:
        sequences.reverse()
    for number in sequences:
        sequence = root / f"{number:02d}"
        (sequence / "velodyne").mkdir(parents=True)
        (sequence / "times.txt").write_bytes(b"0\n0.1\n0.2\n")
        for frame in (reversed(range(3)) if reverse else range(3)):
            data = struct.pack("<ffff", float(number), float(frame), 0.0, 1.0)
            (sequence / "velodyne" / f"{frame:06d}.bin").write_bytes(data)


class ManifestIdentity(unittest.TestCase):
    def test_relocation_and_creation_order_do_not_change_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first", root / "elsewhere" / "second"
            make_dataset(first)
            make_dataset(second, reverse=True)
            a, b = BUILDER.build(first), BUILDER.build(second)

            self.assertNotEqual(a["root"], b["root"])
            self.assertTrue(Path(a["root"]).is_absolute())
            self.assertEqual(a["dataset_sha256"], b["dataset_sha256"])
            self.assertEqual(a["frames"], b["frames"])
            self.assertEqual(a["input_files"], b["input_files"])
            for key in ("frames", "input_files"):
                paths = [record["path"] for record in a[key]]
                self.assertEqual(paths, sorted(paths))
                self.assertTrue(all(not Path(path).is_absolute() for path in paths))

            canonical = {key: value for key, value in a.items()
                         if key not in ("root", "dataset_sha256")}
            payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
            self.assertEqual(a["dataset_sha256"], hashlib.sha256(payload.encode()).hexdigest())
            self.assertEqual(a["dataset_sha256"], BUILDER.dataset_sha256(a))

    def test_same_length_changed_bytes_change_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root)
            before = BUILDER.build(root)
            frame = root / "11/velodyne/000000.bin"
            data = bytearray(frame.read_bytes())
            data[0] ^= 1
            frame.write_bytes(data)
            after = BUILDER.build(root)

            self.assertEqual(before["frames"][0]["bytes"], after["frames"][0]["bytes"])
            self.assertNotEqual(before["frames"][0]["sha256"], after["frames"][0]["sha256"])
            self.assertEqual(after["frames"][0]["sha256"], hashlib.sha256(data).hexdigest())
            self.assertNotEqual(before["dataset_sha256"], after["dataset_sha256"])

    def test_all_inputs_including_unselected_frames_are_fingerprinted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root)
            before = BUILDER.build(root, sample_every=20)
            frame = root / "11/velodyne/000001.bin"
            self.assertNotIn(frame.relative_to(root).as_posix(),
                             [row["path"] for row in before["frames"]])
            frame.write_bytes(b"\x01" * 16)
            after = BUILDER.build(root, sample_every=20)
            self.assertEqual(before["frames"], after["frames"])
            self.assertNotEqual(before["dataset_sha256"], after["dataset_sha256"])
            self.assertEqual(len(after["input_files"]), 33 + 11)
            for record in after["input_files"]:
                self.assertEqual(record["sha256"],
                                 hashlib.sha256((root / record["path"]).read_bytes()).hexdigest())

    def test_timestamp_metadata_is_fingerprinted_not_assumed_official(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root)
            before = BUILDER.build(root)
            (root / "11/times.txt").write_bytes(b"0\n0.1\n0.3\n")
            after = BUILDER.build(root)
            self.assertNotEqual(before["dataset_sha256"], after["dataset_sha256"])
            self.assertFalse(after["training_split_verified"])

    def test_every_twentieth_frame_is_selected_globally(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root, reverse=True)
            full = BUILDER.build(root)
            all_paths = [frame["path"] for frame in full["frames"]]
            self.assertEqual(full["available_frames"], 33)
            self.assertEqual(full["selected_frames"], 33)
            self.assertEqual(full["sampling"]["sample_every"], 1)
            self.assertEqual(full["sampling"]["sample_offset"], 0)
            for offset in (0, 1, 19):
                with self.subTest(offset=offset):
                    sampled = BUILDER.build(root, sample_every=20, sample_offset=offset)
                    self.assertEqual(sampled["frames"], full["frames"][offset::20])
                    self.assertEqual([frame["path"] for frame in sampled["frames"]], all_paths[offset::20])
                    self.assertEqual(sampled["available_frames"], 33)
                    self.assertEqual(sampled["selected_frames"], len(all_paths[offset::20]))
                    self.assertEqual(sampled["sampling"], dict(
                        sample_every=20, sample_offset=offset,
                        order="global-lexicographic-relative-path", index_base=0,
                        selection="frames[sample_offset::sample_every]",
                    ))
            self.assertEqual(all_paths[::20], ["11/velodyne/000000.bin", "17/velodyne/000002.bin"])

    def test_sampling_semantics_are_part_of_identity_even_when_selection_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root)
            a = BUILDER.build(root, sample_every=100)
            b = BUILDER.build(root, sample_every=101)
            self.assertEqual(a["frames"], b["frames"])
            self.assertNotEqual(a["dataset_sha256"], b["dataset_sha256"])

    def test_split_is_declared_not_a_claim_of_verified_training_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root)
            manifest = BUILDER.build(root)
            self.assertEqual(manifest["training_sequences"], [f"{n:02d}" for n in range(11)])
            self.assertEqual(manifest["evaluation_sequences"], [f"{n:02d}" for n in range(11, 22)])
            self.assertFalse(manifest["training_split_verified"])
            self.assertTrue(set(manifest["training_sequences"]).isdisjoint(manifest["evaluation_sequences"]))
            self.assertEqual({row["sequence"] for row in manifest["frames"]},
                             set(manifest["evaluation_sequences"]))
            training = root / "00/velodyne"
            training.mkdir(parents=True)
            (training / "000000.bin").write_bytes(bytes(16))
            self.assertEqual(manifest["dataset_sha256"], BUILDER.build(root)["dataset_sha256"])

    def test_sampling_does_not_skip_validation_of_unselected_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root)
            (root / "21/velodyne/000002.bin").unlink()
            with self.assertRaisesRegex(ValueError, "does not match"):
                BUILDER.build(root, sample_every=20)

    def test_invalid_sampling_arguments_are_rejected(self):
        for stride, offset in ((0, 0), (-1, 0), (1, -1), (20, 20), (1.5, 0), (2, 0.5)):
            with self.subTest(stride=stride, offset=offset):
                with self.assertRaises(ValueError):
                    BUILDER.build("unused", sample_every=stride, sample_offset=offset)

    def test_cli_reports_portable_identity_and_exact_output_file_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sequences"
            make_dataset(root)
            output = Path(directory) / "manifest.json"
            command = [sys.executable, str(BUILDER_PATH), "--sequences", str(root),
                       "--output", str(output), "--sample-every", "20", "--sample-offset", "1"]
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            summary = json.loads(result.stdout)
            manifest = json.loads(output.read_text())
            self.assertEqual(summary["dataset_sha256"], manifest["dataset_sha256"])
            self.assertEqual(summary["manifest_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(summary["frames"], 2)
            self.assertEqual(summary["available_frames"], 33)
            self.assertEqual(summary["sampling"], manifest["sampling"])
            self.assertEqual(manifest["frames"], BUILDER.build(root)["frames"][1::20])
            # An existing manifest must never be silently replaced.
            original = output.read_bytes()
            rerun = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
