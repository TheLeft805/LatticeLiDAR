"""Decode using only a stream and shared weights; no source point cloud is read."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from cli_paths import require_new_output, staged_output, unsigned_u32
from model_config import add_model_arguments, create_codec
from lidar_codec.io import write_points
from lidar_codec.sequence import LEGACY, PREFIX, SequenceReader


def positive_frame_limit(value):
    try:
        limit = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Frame byte limit must be a positive integer"
        ) from None
    if not 0 < limit <= sys.maxsize - PREFIX - LEGACY.size - 1:
        raise argparse.ArgumentTypeError(
            "Frame byte limit must be a positive, supported integer"
        )
    return limit


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
    parser.add_argument("--sequence", type=unsigned_u32, default=0)
    parser.add_argument(
        "--max-frame-bytes",
        type=positive_frame_limit,
        default=64 * 1024 * 1024,
        help=(
            "Maximum encoded body bytes per frame, excluding the 340-byte "
            "identity/size/checksum wrapper (default: 64 MiB)"
        ),
    )
    args = parser.parse_args()
    require_new_output(args.output)
    with args.input.open("rb") as stream:
        sequence = stream.read(4) == b"LSQ2"
        stream.seek(0)
        if not sequence and args.output.suffix not in (".npy", ".ply"):
            raise ValueError("Reconstruction must be .npy or .ply")
        if not sequence:
            packet_limit = args.max_frame_bytes + PREFIX + LEGACY.size
            packet = stream.read(packet_limit + 1)
            if len(packet) > packet_limit:
                raise ValueError("Encoded frame exceeds --max-frame-bytes")
        with staged_output(args.output, directory=sequence) as temporary:
            codec = create_codec(args, decoding=True)
            try:
                if sequence:
                    reader = SequenceReader(
                        hashlib.sha256(codec.prefix).hexdigest(),
                        args.sequence,
                        max_body_bytes=args.max_frame_bytes,
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
                                temporary / f"{number:06d}.ply",
                                points,
                            )
                            count += 1
                    reader.finish()
                    result = dict(frames=count, complete=True)
                else:
                    start = time.perf_counter()
                    points, stats = codec.decode(packet)
                    elapsed = time.perf_counter() - start
                    write_points(temporary, points)
                    result = dict(
                        output=str(args.output), points=len(points), seconds=elapsed
                    )
            finally:
                codec.close()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
