"""Validate and fingerprint SemanticKITTI sequences 11..21, optionally sampling.

Sampling applies once to the global, lexicographically sorted relative frame
paths: frames[sample_offset::sample_every]. The default includes every frame.
"""

import argparse
import hashlib
import json
from pathlib import Path


def _file_record(path, root):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return dict(
        path=path.relative_to(root).as_posix(),
        bytes=size,
        sha256=digest.hexdigest(),
    )


def dataset_sha256(manifest):
    """Hash the canonical manifest, excluding only local root and this digest.

    Input hashes cover all frame and timestamp files, including unselected
    frames. The selected frame list and sampling semantics are also bound to
    the identity. JSON uses sorted keys, compact separators and UTF-8 encoding.
    """
    canonical = {
        key: value
        for key, value in manifest.items()
        if key not in ("root", "dataset_sha256")
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build(root, sample_every=1, sample_offset=0):
    if type(sample_every) is not int or sample_every < 1:
        raise ValueError("sample_every must be a positive integer")
    if type(sample_offset) is not int or not 0 <= sample_offset < sample_every:
        raise ValueError("sample_offset must be an integer in [0, sample_every)")
    root = Path(root).resolve()
    frames = []
    input_files = []
    for sequence in range(11, 22):
        name = f"{sequence:02d}"
        files = sorted((root / name / "velodyne").glob("*.bin"))
        if not files:
            raise FileNotFoundError(f"Missing or empty test sequence {name}")
        numbers = [int(path.stem) for path in files]
        timestamps = root / name / "times.txt"
        if not timestamps.is_file():
            raise FileNotFoundError(
                f"Missing timestamp metadata: {timestamps}"
            )
        timestamp_bytes = timestamps.read_bytes()
        expected = len(timestamp_bytes.decode("utf-8").splitlines())
        input_files.append(
            dict(
                path=timestamps.relative_to(root).as_posix(),
                bytes=len(timestamp_bytes),
                sha256=hashlib.sha256(timestamp_bytes).hexdigest(),
            )
        )
        if numbers != list(range(expected)):
            raise ValueError(
                f"Sequence {name} does not match its {expected} timestamps"
            )
        for path in files:
            record = _file_record(path, root)
            size = record["bytes"]
            if not size or size % 16:
                raise ValueError(f"Invalid KITTI frame size: {path}")
            input_files.append(record)
            frames.append(
                dict(
                    sequence=name,
                    frame=int(path.stem),
                    points=size // 16,
                    **record,
                )
            )
    frames.sort(key=lambda frame: frame["path"])
    selected = frames[sample_offset::sample_every]
    manifest = dict(
        schema="semantickitti-test-v3",
        split="11-21",
        training_sequences=[f"{number:02d}" for number in range(11)],
        evaluation_sequences=[f"{number:02d}" for number in range(11, 22)],
        # These are split declarations, not proof of a model's training inputs.
        training_split_verified=False,
        sampling=dict(
            sample_every=sample_every,
            sample_offset=sample_offset,
            order="global-lexicographic-relative-path",
            index_base=0,
            selection="frames[sample_offset::sample_every]",
        ),
        available_frames=len(frames),
        selected_frames=len(selected),
        input_files=sorted(input_files, key=lambda record: record["path"]),
        root=str(root),
        frames=selected,
    )
    manifest["dataset_sha256"] = dataset_sha256(manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--sample-every", type=int, default=1,
        help="Select every Nth frame from global sorted order (default: 1, all).",
    )
    parser.add_argument(
        "--sample-offset", type=int, default=0,
        help="Zero-based starting index, in [0, sample-every) (default: 0).",
    )
    args = parser.parse_args()
    if args.sample_every < 1 or not 0 <= args.sample_offset < args.sample_every:
        parser.error("--sample-every must be positive and 0 <= --sample-offset < --sample-every")
    value = build(args.sequences, args.sample_every, args.sample_offset)
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(
        json.dumps(
            dict(
                frames=len(value["frames"]),
                available_frames=value["available_frames"],
                sampling=value["sampling"],
                dataset_sha256=value["dataset_sha256"],
                # This is the exact output-file checksum, including local root.
                manifest_sha256=hashlib.sha256(payload.encode()).hexdigest(),
            )
        )
    )


if __name__ == "__main__":
    main()
