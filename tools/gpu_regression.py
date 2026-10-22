"""GPU regression worker; run encoding and decoding in separate processes.

Each worker visits every frame once, then performs --repeat additional replays.
Encoding requires an empty output directory. Its completed encode.json is the
manifest for decoding, so incomplete runs and unrelated artifacts are rejected.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


RUN_SCHEMA_VERSION = 1


def iteration_plan(frame_count, repeats):
    """Cover all frames, then replay frames in order (not repeats per frame)."""
    if frame_count < 1:
        raise ValueError("At least one input frame is required")
    if repeats < 2:
        raise ValueError("At least two additional replays are required")
    return tuple(range(frame_count)) + tuple(index % frame_count for index in range(repeats))


def packet_files(output):
    """Sort numeric packet names numerically, including indices above 999."""
    files = list(output.glob("*.lgr"))
    if any(not path.stem.isascii() or not path.stem.isdecimal() for path in files):
        raise ValueError("Regression packet filenames must be numeric")
    indices = [int(path.stem) for path in files]
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate numeric packet indices")
    return sorted(files, key=lambda path: int(path.stem))


def reference_path(packet):
    """References follow the actual packet path, not its position in a list."""
    return packet.with_suffix(".expected.npy")


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prepare_encode_output(output):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Encode output must be empty; use a new directory for each regression run")


def load_encode_run(output, quantization_mm):
    """Validate completed encoder artifacts before loading the GPU codec.

    Decoder artifacts for this exact frame set may be overwritten on a rerun.
    They are never used as inputs, and the previous decode.json is invalidated
    before decoding starts. Unexpected artifacts, even extra references, fail.
    """
    manifest_path = output / "encode.json"
    if not manifest_path.is_file():
        raise ValueError("Missing completed encode.json; rerun encoding in a new directory")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("Unsupported regression manifest; rerun encoding in a new directory")
    if manifest.get("mode") != "encode" or manifest.get("passed") is not True:
        raise ValueError("Encoding did not complete successfully")
    if manifest.get("quantization_mm") != quantization_mm:
        raise ValueError("Regression quantization does not match the encoded run")
    files = packet_files(output)
    artifacts = manifest.get("artifacts")
    count = len(files)
    if not count or not isinstance(artifacts, list) or len(artifacts) != count:
        raise ValueError("Encoded packet set does not match the manifest")
    if manifest.get("frames") != count or manifest.get("supplied_frames") != count:
        raise ValueError("Encoder manifest does not record complete frame coverage")
    repeats = manifest.get("repeats")
    if (type(repeats) is not int or repeats < 2
            or manifest.get("verified_repeats") != repeats
            or manifest.get("iterations") != count + repeats):
        raise ValueError("Encoder manifest does not record all checked replays")
    allowed = {"encode.json", "decode.json", "cdf.json"}
    for index, (packet, artifact) in enumerate(zip(files, artifacts)):
        if (packet.name != f"{index:03d}.lgr" or not isinstance(artifact, dict)
                or artifact.get("packet") != packet.name):
            raise ValueError("Encoded packet names do not match the manifest")
        expected = reference_path(packet)
        allowed.update((packet.name, expected.name, packet.with_suffix(".xyz.npy").name))
        for path, field in ((packet, "packet_sha256"), (expected, "expected_sha256")):
            if not path.is_file() or file_digest(path) != artifact.get(field):
                raise ValueError(f"Missing or changed regression artifact: {path.name}")
    trace = output / "cdf.json"
    if not trace.is_file() or file_digest(trace) != manifest.get("cdf_sha256"):
        raise ValueError("Missing or changed regression artifact: cdf.json")
    unexpected = sorted(path.name for path in output.iterdir() if path.name not in allowed)
    if unexpected:
        raise ValueError(f"Unexpected regression artifacts: {', '.join(unexpected)}")
    return manifest, files


def check(condition, message):
    """Keep essential regression validation active under python -O."""
    if not condition:
        raise RuntimeError(message)


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
    parser.add_argument("--repeat", type=int, default=12,
                        help="additional checked replays after visiting every frame once (default: 12)")
    parser.add_argument("--bounded-records", action="store_true")
    args = parser.parse_args()
    if args.repeat < 2 or (args.mode == "encode" and not args.input):
        parser.error("At least two additional replays and encode input frames are required")
    output = args.output
    manifest = None
    if args.mode == "encode":
        prepare_encode_output(output)
        files = args.input
    else:
        # Even a failed preflight must not leave an earlier successful report.
        (output / "decode.json").unlink(missing_ok=True)
        manifest, files = load_encode_run(output, args.quantization_mm)
    plan = iteration_plan(len(files), args.repeat)
    root = args.codec_root.resolve()
    sys.path.insert(0, str(root))
    from trusted_assets import verify_release_assets
    verify_release_assets(root / "weights/backbone.pt", root / "weights/reconstruction.npz")
    import torch
    from reconstruction_codec import ReconstructionCodec
    from lidar_codec.geometry import quantize
    from lidar_codec.io import read_points

    expected_profile = json.loads((root / f"configs/q{args.quantization_mm}.json").read_text())
    codec = ReconstructionCodec(root / "weights/backbone.pt", root / "weights/reconstruction.npz",
                                kind="gelu64", quantization_mm=args.quantization_mm,
                                profile=None if args.default_profile else expected_profile)
    rows = []
    references = {}
    artifacts = []
    verified_repeats = 0
    try:
        check(codec.base.plan == expected_profile, "Codec profile differs from the release profile")
        if manifest is not None:
            check(manifest.get("source_sha256") == codec.base.identity["source_sha256"],
                  "Codec source identity differs from the encoded run")
        codec.base.execution.select("combined" if args.mode == "decode" else "baseline")
        for iteration, index in enumerate(plan):
            if args.mode == "encode":
                packet_path = output / f"{index:03d}.lgr"
                raw = read_points(files[index])
                packet, _ = codec.encode(raw)
                value = hashlib.sha256(packet).hexdigest()
                if index not in references:
                    packet_path.write_bytes(packet)
                    expected = np.unique(quantize(raw, codec.geometry), axis=0)
                    np.save(reference_path(packet_path), expected)
                    artifacts.append(dict(packet=packet_path.name, packet_sha256=value,
                                          expected_sha256=file_digest(reference_path(packet_path))))
                if iteration == 0:
                    traced, stats = codec.base.encode(raw, trace=True)
                    check(traced[len(codec.prefix):] == packet[len(codec.prefix):],
                          "Traced encoding differs from the encoded packet")
                    (output / "cdf.json").write_text(json.dumps(dict(sha256=digest_trace(stats["trace"]))))
                    del stats
            else:
                packet_path = files[index]
                packet = packet_path.read_bytes()
                points, _ = codec.decode(packet)
                quantized = codec.last_quantized
                np.testing.assert_array_equal(np.unique(quantized, axis=0),
                                              np.load(reference_path(packet_path), allow_pickle=False))
                value = hashlib.sha256(points.tobytes()).hexdigest()
                if index not in references:
                    np.save(packet_path.with_suffix(".xyz.npy"), points)
                if iteration == 0:
                    _, stats = codec.base.decode(codec.base.prefix + packet[len(codec.prefix):], trace=True)
                    expected_trace = json.loads((output / "cdf.json").read_text())["sha256"]
                    check(digest_trace(stats["trace"]) == expected_trace,
                          "Decoder trace differs from the encoder trace")
                    del stats
            if index in references:
                check(references[index] == value, "Cross-frame reuse changed the result")
                verified_repeats += 1
            references[index] = value
            records = codec.base.stage.records
            if args.bounded_records:
                check(len(records) <= 21, "Stage records exceeded the per-frame bound")
                check(len({r["frame_index"] for r in records}) == 1,
                      "Stage records do not belong to exactly one frame")
            rss = int(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmRSS:")))
            rows.append(dict(iteration=iteration, frame=index, records=len(records), rss_kib=rss,
                             cuda_allocated=torch.cuda.memory_allocated(), sha256=value))
        check(len(references) == len(files), "Not every supplied frame was tested")
        check(verified_repeats == args.repeat, "Not every requested replay was verified")
        if args.mode == "decode":
            bad = bytearray(packet)
            bad[-1] ^= 1
            try:
                codec.decode(bytes(bad))
            except ValueError:
                pass
            else:
                raise RuntimeError("Corruption was accepted")
            recovered, _ = codec.decode(packet)
            check(hashlib.sha256(recovered.tobytes()).hexdigest() == value,
                  "Decode did not recover after rejecting corruption")
        result = dict(schema_version=RUN_SCHEMA_VERSION, mode=args.mode, quantization_mm=args.quantization_mm,
                      default_profile=args.default_profile, frames=len(references), supplied_frames=len(files),
                      iterations=len(rows), repeats=args.repeat, verified_repeats=verified_repeats,
                      source_sha256=codec.base.identity["source_sha256"], rows=rows, passed=True)
        if args.mode == "encode":
            result.update(artifacts=artifacts, cdf_sha256=file_digest(output / "cdf.json"))
        (output / f"{args.mode}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "artifacts")}), flush=True)
    finally:
        codec.close()


if __name__ == "__main__":
    main()
