"""Verify the released numerical sources, profiles and weights without CUDA."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / "ASSETS.json").read_text(encoding="utf-8"))
    failures = []
    for name, expected in manifest["files"].items():
        path = ROOT / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            failures.append(name)
    if failures:
        raise SystemExit("Missing or changed release assets: " + ", ".join(failures))
    print(f"Verified {len(manifest['files'])} frozen files (SHA-256).")


if __name__ == "__main__":
    verify()
