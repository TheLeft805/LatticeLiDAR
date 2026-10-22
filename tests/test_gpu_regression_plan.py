"""CPU-only checks for regression scheduling and artifact isolation, not GPU validation."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import gpu_regression as regression


class RegressionPlan(unittest.TestCase):
    def test_full_coverage_precedes_twelve_verified_replays(self):
        for count in (1, 3, 15, 386, 1018):
            with self.subTest(frames=count):
                plan = regression.iteration_plan(count, 12)
                self.assertEqual(plan[:count], tuple(range(count)))
                self.assertEqual(len(plan), count + 12)
                seen = set()
                verified_repeats = 0
                for frame in plan:
                    if frame in seen:
                        verified_repeats += 1
                    seen.add(frame)
                self.assertEqual(len(seen), count)
                self.assertEqual(verified_repeats, 12)
                self.assertEqual(plan[count:], tuple(index % count for index in range(12)))

    def test_small_frame_sets_keep_previous_schedule(self):
        for count in (1, 2, 3):
            expected = tuple(index % count for index in range(12 + count))
            self.assertEqual(regression.iteration_plan(count, 12), expected)

    def test_invalid_plans_fail(self):
        for count, repeats in ((0, 12), (-1, 12), (15, 1), (15, -1)):
            with self.subTest(frames=count, repeats=repeats):
                with self.assertRaises(ValueError):
                    regression.iteration_plan(count, repeats)

    def test_numeric_order_and_packet_relative_references_above_999(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("1000.lgr", "999.lgr", "001.lgr", "1001.lgr"):
                (root / name).touch()
            packets = regression.packet_files(root)
            self.assertEqual([path.name for path in packets],
                             ["001.lgr", "999.lgr", "1000.lgr", "1001.lgr"])
            self.assertEqual(regression.reference_path(packets[2]), root / "1000.expected.npy")

    def test_duplicate_or_nonnumeric_packet_names_fail(self):
        for names in (("001.lgr", "1.lgr"), ("old.lgr",)):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in names:
                    (root / name).touch()
                with self.assertRaises(ValueError):
                    regression.packet_files(root)

    def test_essential_validation_survives_python_optimization(self):
        root = Path(__file__).parents[1]
        result = subprocess.run(
            [sys.executable, "-O", "-c",
             "from tools.gpu_regression import check; check(False, 'required regression check')"],
            cwd=root, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required regression check", result.stderr)


class RegressionArtifacts(unittest.TestCase):
    def make_run(self, root, count=3):
        """Build only a filesystem manifest fixture; no codec or GPU is mocked."""
        artifacts = []
        for index in range(count):
            packet = root / f"{index:03d}.lgr"
            packet.write_bytes(f"packet {index}".encode())
            expected = regression.reference_path(packet)
            expected.write_bytes(f"reference {index}".encode())
            artifacts.append(dict(packet=packet.name, packet_sha256=regression.file_digest(packet),
                                  expected_sha256=regression.file_digest(expected)))
        (root / "cdf.json").write_text('{"sha256": "trace fixture"}')
        manifest = dict(schema_version=regression.RUN_SCHEMA_VERSION, mode="encode", passed=True,
                        quantization_mm=16, frames=count, supplied_frames=count,
                        repeats=12, verified_repeats=12, iterations=count + 12,
                        artifacts=artifacts, cdf_sha256=regression.file_digest(root / "cdf.json"))
        (root / "encode.json").write_text(json.dumps(manifest))
        return manifest

    def test_encode_accepts_empty_directory_and_rejects_stale_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            regression.prepare_encode_output(output)
            (output / "000.lgr").write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "must be empty"):
                regression.prepare_encode_output(output)

    def test_manifest_round_trip_has_full_numeric_order(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            expected = self.make_run(output, count=1018)
            manifest, packets = regression.load_encode_run(output, 16)
            self.assertEqual(manifest, expected)
            self.assertEqual(len(packets), 1018)
            self.assertEqual(packets[1000].name, "1000.lgr")
            self.assertEqual(regression.reference_path(packets[1000]).name, "1000.expected.npy")

    def test_completed_decode_artifacts_can_be_overwritten_but_are_not_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.make_run(output)
            (output / "decode.json").write_text('{"passed": true}')
            (output / "000.xyz.npy").write_bytes(b"previous decoder output")
            _, packets = regression.load_encode_run(output, 16)
            self.assertEqual([path.name for path in packets], ["000.lgr", "001.lgr", "002.lgr"])

    def test_incomplete_or_legacy_encoding_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, "Missing completed"):
                regression.load_encode_run(output, 16)
            (output / "encode.json").write_text('{"passed": true}')
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                regression.load_encode_run(output, 16)

    def test_missing_changed_or_extra_artifacts_are_rejected(self):
        cases = ("missing packet", "missing reference", "changed packet", "changed reference",
                 "changed trace", "extra packet", "extra reference", "extra decoder output")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                self.make_run(output)
                if case == "missing packet":
                    (output / "001.lgr").unlink()
                elif case == "missing reference":
                    (output / "001.expected.npy").unlink()
                elif case == "changed packet":
                    (output / "001.lgr").write_bytes(b"stale")
                elif case == "changed reference":
                    (output / "001.expected.npy").write_bytes(b"stale")
                elif case == "changed trace":
                    (output / "cdf.json").write_text("{}")
                elif case == "extra packet":
                    (output / "003.lgr").write_bytes(b"stale")
                elif case == "extra reference":
                    (output / "003.expected.npy").write_bytes(b"stale")
                else:
                    (output / "003.xyz.npy").write_bytes(b"stale")
                with self.assertRaises(ValueError):
                    regression.load_encode_run(output, 16)

    def test_failed_decode_preflight_invalidates_previous_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            previous = output / "decode.json"
            previous.write_text('{"passed": true}')
            argv = ["gpu_regression.py", "--codec-root", str(Path(__file__).parents[1]),
                    "--output", str(output), "--mode", "decode", "--quantization-mm", "16"]
            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "Missing completed"):
                    regression.main()
            self.assertFalse(previous.exists())

    def test_assets_are_checked_before_importing_gpu_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(__file__).parents[1]
            argv = ["gpu_regression.py", "--codec-root", str(root),
                    "--output", directory, "--mode", "encode", "--input", "frame.bin",
                    "--quantization-mm", "16"]
            # Exercise only the preflight rejection, not a simulated GPU run.
            with patch.object(sys, "argv", argv), patch.object(sys, "path", list(sys.path)):
                with patch.dict(sys.modules, {"torch": None, "reconstruction_codec": None}):
                    with patch("trusted_assets.verify_release_assets", side_effect=ValueError("untrusted")) as verify:
                        with self.assertRaisesRegex(ValueError, "untrusted"):
                            regression.main()
                        verify.assert_called_once_with(root / "weights/backbone.pt",
                                                       root / "weights/reconstruction.npz")

    def test_incomplete_coverage_and_replays_or_wrong_quantization_are_rejected(self):
        for field, value in (("frames", 2), ("supplied_frames", 4), ("verified_repeats", 0),
                             ("iterations", 12), ("repeats", 1), ("passed", False),
                             ("mode", "decode"), ("quantization_mm", 32)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                manifest = self.make_run(output)
                manifest[field] = value
                (output / "encode.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    regression.load_encode_run(output, 16)


if __name__ == "__main__":
    unittest.main()
