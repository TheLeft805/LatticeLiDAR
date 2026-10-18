"""Enumerate every frame of SemanticKITTI test sequences 11..21, without sampling."""

import argparse
import hashlib
import json
from pathlib import Path


def build(root):
    root = Path(root).resolve()
    frames = []
    for sequence in range(11, 22):
        name = f"{sequence:02d}"
        files = sorted((root / name / "velodyne").glob("*.bin"))
        if not files:
            raise FileNotFoundError(f"Missing or empty test sequence {name}")
        numbers = [int(path.stem) for path in files]
        timestamps = root / name / "times.txt"
        if not timestamps.is_file():
            raise FileNotFoundError(
                f"Missing official timestamp metadata: {timestamps}"
            )
        expected = len(timestamps.read_text().splitlines())
        if numbers != list(range(expected)):
            raise ValueError(
                f"Sequence {name} does not match its {expected} official timestamps"
            )
        for path in files:
            size = path.stat().st_size
            if not size or size % 16:
                raise ValueError(f"Invalid KITTI frame size: {path}")
            frames.append(
                dict(
                    sequence=name,
                    frame=int(path.stem),
                    path=path.relative_to(root).as_posix(),
                    bytes=size,
                    points=size // 16,
                )
            )
    return dict(
        schema="semantickitti-full-test-v1",
        split="11-21",
        sampling="none",
        root=str(root),
        frames=frames,
        exposure=dict(
            development=["13", "15", "19"],
            review=["14", "18", "21"],
            freeze_before_test=["12", "16", "20"],
            additional=["11", "17"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = build(args.sequences)
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(
        json.dumps(
            dict(
                frames=len(value["frames"]),
                manifest_sha256=hashlib.sha256(payload.encode()).hexdigest(),
            )
        )
    )


if __name__ == "__main__":
    main()
