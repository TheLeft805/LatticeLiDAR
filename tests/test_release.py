import argparse
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

import model_config


class ReleaseContract(unittest.TestCase):
    def test_all_scales_bind_final_head_and_endpoint_policy(self):
        for q in model_config.QUANTIZATION_MM:
            for decoding, policy in ((False, "baseline"), (True, "combined")):
                parser = argparse.ArgumentParser()
                model_config.add_model_arguments(parser)
                args = parser.parse_args(["--quantization-mm", str(q)])
                constructor = Mock()
                module = types.SimpleNamespace(ReconstructionCodec=constructor)
                with patch.dict(sys.modules, {"reconstruction_codec": module}):
                    model_config.create_codec(args, decoding=decoding)
                kwargs = constructor.call_args.kwargs
                self.assertEqual(kwargs["kind"], "gelu64")
                self.assertEqual(kwargs["quantization_mm"], q)
                expected = json.loads((model_config.ROOT / f"configs/q{q}.json").read_text())
                self.assertEqual(kwargs["profile"], expected)
                constructor.return_value.base.execution.select.assert_called_once_with(policy)

    def test_decoder_writes_reconstructed_meters_without_second_transform(self):
        import decode

        points = np.array([[1.234, -2.0, 0.125]], dtype=np.float32)
        codec = Mock()
        codec.decode.return_value = (points, {})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "frame.lgr", root / "frame.npy"
            source.write_bytes(b"LGR1fixture")
            argv = ["decode.py", "--input", str(source), "--output", str(output)]
            with patch.object(sys, "argv", argv), patch.object(decode, "create_codec", return_value=codec):
                decode.main()
            np.testing.assert_array_equal(np.load(output), points)
            codec.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
