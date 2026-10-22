"""Preflight checks for the frozen release's CLI/model entry points.

ASSETS.json must come from a trusted repository checkout. This is an allowlist,
not a replacement for patched deserialization libraries or a sandbox. Direct
research use of GeometryCodec/ReconstructionCodec is outside this boundary.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release_assets(checkpoint, heads, *, manifest_path=None):
    """Reject non-release assets before importing or constructing a GPU model.

    Identical copies at other paths are accepted. Keep these files and the
    trusted checkout unwritable by other users throughout model construction.
    """
    manifest_path = ROOT / "ASSETS.json" if manifest_path is None else Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    for path, name in ((checkpoint, "weights/backbone.pt"),
                       (heads, "weights/reconstruction.npz")):
        expected = files.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Missing trusted release digest for {name}")
        if file_sha256(path) != expected:
            raise ValueError(
                f"Untrusted or modified release asset: {path}. "
                "Use the published assets from a trusted checkout; custom checkpoints "
                "are not supported by the release CLI."
            )
