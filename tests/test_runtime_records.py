"""CPU regression for the stage scheduler's host-side record lifetime."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


def load_scheduler_method():
    source = Path(__file__).parents[1] / "lidar_codec/stage.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "StageExecutor")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "predictions")
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["predictions"]


class RecordLifetime(unittest.TestCase):
    def test_repeated_frames_and_failure_do_not_accumulate_records(self):
        predictions = load_scheduler_method()
        bucket = SimpleNamespace(capacity=32, posQ=16, predictions=lambda item: iter(()))
        stage = SimpleNamespace(frame_index=-1, records=[], max_rows={}, buckets={1: [bucket], 2: [bucket]})
        first_frame = None
        for frame in range(386):
            for depth, scale in enumerate((2, 1)):
                list(predictions(stage, [0] * 8, 0, None, scale, depth, scale == 1, posQ=16))
            self.assertEqual(len(stage.records), 2)
            self.assertTrue(all(row["frame_index"] == frame for row in stage.records))
            if first_frame is None:
                first_frame = stage.records
        self.assertEqual([row["frame_index"] for row in first_frame], [0, 0])
        with self.assertRaises(ValueError):
            list(predictions(stage, [0] * 33, 0, None, 2, 0, False, posQ=16))
        self.assertEqual(stage.records, [])
        list(predictions(stage, [0], 0, None, 1, 0, True, posQ=16))
        self.assertEqual(len(stage.records), 1)


if __name__ == "__main__":
    unittest.main()
