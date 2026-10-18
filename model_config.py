"""The released model: matched weights, reconstruction and execution profiles."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUANTIZATION_MM = (8, 16, 32, 64)


def add_model_arguments(parser):
    parser.add_argument("--quantization-mm", type=int, choices=QUANTIZATION_MM, default=16)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/backbone.pt")
    parser.add_argument("--heads", type=Path, default=ROOT / "weights/reconstruction.npz")


def create_codec(args, *, decoding=False):
    """Load the frozen GELU head and the matching profile; requires Linux/CUDA."""
    from reconstruction_codec import ReconstructionCodec

    profile = json.loads((ROOT / f"configs/q{args.quantization_mm}.json").read_text())
    codec = ReconstructionCodec(
        args.checkpoint,
        args.heads,
        kind="gelu64",
        quantization_mm=args.quantization_mm,
        profile=profile,
    )
    codec.base.execution.select("combined" if decoding else "baseline")
    return codec
