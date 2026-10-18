"""Encode one KITTI/NPY frame, or a directory of KITTI frames as LSQ2."""

import argparse
import json
from pathlib import Path
import time
from model_config import add_model_arguments, create_codec
from lidar_codec.io import read_points
from lidar_codec.sequence import SequenceWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_model_arguments(parser)
    parser.add_argument("--sequence", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    files = sorted(args.input.glob("*.bin")) if args.input.is_dir() else [args.input]
    if not files:
        raise ValueError("No input frames")
    setup = time.perf_counter()
    codec = create_codec(args, decoding=False)
    setup = time.perf_counter() - setup
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    records = []
    try:
        writer = (
            SequenceWriter(args.sequence, codec.prefix) if args.input.is_dir() else None
        )
        with temporary.open("xb") as stream:
            if writer:
                stream.write(writer.header)
            for filename in files:
                points = read_points(filename)
                start = time.perf_counter()
                packet, stats = codec.encode(points)
                body = (
                    writer.frame(int(filename.stem), True, packet) if writer else packet
                )
                elapsed = time.perf_counter() - start
                stream.write(body)
                records.append(
                    dict(
                        input=filename.name,
                        input_points=len(points),
                        bytes=len(body),
                        seconds=elapsed,
                    )
                )
            if writer:
                stream.write(writer.finish())
        temporary.replace(args.output)
        print(
            json.dumps(
                dict(
                    output=str(args.output),
                    bytes=args.output.stat().st_size,
                    setup_seconds=setup,
                    frames=records,
                    timing_scope="CPU XYZ to CPU frame/sequence record; disk I/O excluded",
                ),
                indent=2,
            )
        )
    finally:
        codec.close()


if __name__ == "__main__":
    main()
