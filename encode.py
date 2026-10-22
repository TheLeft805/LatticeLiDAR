"""Encode one KITTI/NPY frame, or a directory of KITTI frames as LSQ2."""

import argparse
import json
from pathlib import Path
import time
from cli_paths import require_new_output, sequence_inputs, staged_output, unsigned_u32
from model_config import add_model_arguments, create_codec
from lidar_codec.io import read_points
from lidar_codec.sequence import SequenceWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_model_arguments(parser)
    parser.add_argument("--sequence", type=unsigned_u32, default=0)
    args = parser.parse_args()
    require_new_output(args.output)
    sequence = args.input.is_dir()
    files = sequence_inputs(args.input) if sequence else [(None, args.input)]
    records = []
    with staged_output(args.output) as temporary:
        setup = time.perf_counter()
        codec = create_codec(args, decoding=False)
        setup = time.perf_counter() - setup
        try:
            writer = SequenceWriter(args.sequence, codec.prefix) if sequence else None
            with temporary.open("xb") as stream:
                if writer:
                    stream.write(writer.header)
                for number, filename in files:
                    points = read_points(filename)
                    start = time.perf_counter()
                    packet, stats = codec.encode(points)
                    body = writer.frame(number, True, packet) if writer else packet
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
            output_bytes = temporary.stat().st_size
        finally:
            codec.close()
    print(
        json.dumps(
            dict(
                output=str(args.output),
                bytes=output_bytes,
                setup_seconds=setup,
                frames=records,
                timing_scope="CPU XYZ to CPU frame/sequence record; disk I/O excluded",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
