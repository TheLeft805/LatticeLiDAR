"""GPU regression worker; run encoding and decoding in separate processes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def digest_trace(rows):
    result = hashlib.sha256()
    for row in rows:
        for name in ("labels", "base", "components", "priors"):
            value = row[name]
            result.update(name.encode())
            if value is None:
                result.update(b"none")
            else:
                array = np.asarray(value, dtype="<i8")
                result.update(str(array.shape).encode())
                result.update(array.tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("encode", "decode"), required=True)
    parser.add_argument("--input", type=Path, nargs="+")
    parser.add_argument("--quantization-mm", type=int, choices=(8, 16, 32, 64), required=True)
    parser.add_argument("--default-profile", action="store_true")
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--bounded-records", action="store_true")
    args = parser.parse_args()
    if args.repeat < 2 or (args.mode == "encode" and not args.input):
        parser.error("At least two repeats and encode input frames are required")
    root = args.codec_root.resolve()
    sys.path.insert(0, str(root))
    import torch
    from reconstruction_codec import ReconstructionCodec
    from lidar_codec.geometry import quantize
    from lidar_codec.io import read_points

    expected_profile = json.loads((root / f"configs/q{args.quantization_mm}.json").read_text())
    codec = ReconstructionCodec(root / "weights/backbone.pt", root / "weights/reconstruction.npz",
                                kind="gelu64", quantization_mm=args.quantization_mm,
                                profile=None if args.default_profile else expected_profile)
    assert codec.base.plan == expected_profile
    codec.base.execution.select("combined" if args.mode == "decode" else "baseline")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    references = {}
    try:
        files = args.input if args.mode == "encode" else sorted(output.glob("*.lgr"))
        if not files:
            raise ValueError("No frames")
        for iteration in range(args.repeat + min(3, len(files))):
            index = iteration % len(files)
            prefix = output / f"{index:03d}"
            if args.mode == "encode":
                raw = read_points(files[index])
                packet, _ = codec.encode(raw)
                value = hashlib.sha256(packet).hexdigest()
                if index not in references:
                    prefix.with_suffix(".lgr").write_bytes(packet)
                    expected = np.unique(quantize(raw, codec.geometry), axis=0)
                    np.save(prefix.with_suffix(".expected.npy"), expected)
                if iteration == 0:
                    traced, stats = codec.base.encode(raw, trace=True)
                    assert traced[len(codec.prefix):] == packet[len(codec.prefix):]
                    (output / "cdf.json").write_text(json.dumps(dict(sha256=digest_trace(stats["trace"]))))
                    del stats
            else:
                packet = files[index].read_bytes()
                points, _ = codec.decode(packet)
                quantized = codec.last_quantized
                np.testing.assert_array_equal(np.unique(quantized, axis=0), np.load(prefix.with_suffix(".expected.npy")))
                value = hashlib.sha256(points.tobytes()).hexdigest()
                if index not in references:
                    np.save(prefix.with_suffix(".xyz.npy"), points)
                if iteration == 0:
                    _, stats = codec.base.decode(codec.base.prefix + packet[len(codec.prefix):], trace=True)
                    expected_trace = json.loads((output / "cdf.json").read_text())["sha256"]
                    assert digest_trace(stats["trace"]) == expected_trace
                    del stats
            if index in references:
                assert references[index] == value, "Cross-frame reuse changed the result"
            references[index] = value
            records = codec.base.stage.records
            if args.bounded_records:
                assert len(records) <= 21
                assert len({r["frame_index"] for r in records}) == 1
            rss = int(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmRSS:")))
            rows.append(dict(iteration=iteration, frame=index, records=len(records), rss_kib=rss,
                             cuda_allocated=torch.cuda.memory_allocated(), sha256=value))
        if args.mode == "decode":
            bad = bytearray(packet)
            bad[-1] ^= 1
            try:
                codec.decode(bytes(bad))
            except ValueError:
                pass
            else:
                raise AssertionError("Corruption was accepted")
            recovered, _ = codec.decode(packet)
            assert hashlib.sha256(recovered.tobytes()).hexdigest() == value
        result = dict(mode=args.mode, quantization_mm=args.quantization_mm,
                      default_profile=args.default_profile, frames=len(files), repeats=args.repeat,
                      source_sha256=codec.base.identity["source_sha256"], rows=rows, passed=True)
        (output / f"{args.mode}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
    finally:
        codec.close()


if __name__ == "__main__":
    main()
