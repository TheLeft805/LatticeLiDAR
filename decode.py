"""Decode using only a stream and shared weights; no source point cloud is read."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from model_config import add_model_arguments, create_codec
from lidar_codec.io import write_points
from lidar_codec.sequence import SequenceReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="PLY/NPY for a frame; output directory for LSQ2",
    )
    add_model_arguments(parser)
    parser.add_argument("--sequence", type=int, default=0)
    args = parser.parse_args()
    codec = create_codec(args, decoding=True)
    try:
        with args.input.open("rb") as stream:
            sequence = stream.read(4) == b"LSQ2"
            stream.seek(0)
            if sequence:
                if args.output.exists():
                    raise FileExistsError(args.output)
                reader = SequenceReader(
                    hashlib.sha256(codec.prefix).hexdigest(), args.sequence
                )
                count = 0
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    for number, reset, packet in reader.feed(chunk):
                        if not reset:
                            raise ValueError(
                                "This independent-frame model requires reset records"
                            )
                        points, _ = codec.decode(packet)
                        write_points(
                            args.output / f"{number:06d}.ply",
                            points,
                        )
                        count += 1
                reader.finish()
                print(json.dumps(dict(frames=count, complete=True)))
            else:
                packet = stream.read()
                start = time.perf_counter()
                points, stats = codec.decode(packet)
                reconstructed = points
                elapsed = time.perf_counter() - start
                write_points(args.output, reconstructed)
                print(
                    json.dumps(
                        dict(
                            output=str(args.output), points=len(points), seconds=elapsed
                        )
                    )
                )
    finally:
        codec.close()


if __name__ == "__main__":
    main()
