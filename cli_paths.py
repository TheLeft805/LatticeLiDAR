"""CLI path validation and fail-closed, no-clobber output transactions.

Keep these helpers outside lidar_codec: CLI I/O does not change codec identity.
"""

import argparse
from contextlib import contextmanager
import ctypes
import errno
import os
from pathlib import Path
import tempfile


def unsigned_u32(value):
    """Parse a CLI sequence identifier before constructing a GPU codec."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Sequence ID must be an unsigned 32-bit integer"
        ) from None
    if not 0 <= number <= 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("Sequence ID must be an unsigned 32-bit integer")
    return number


def sequence_inputs(directory):
    """Return (frame ID, path) pairs in numeric, not lexicographic, order."""
    frames = []
    seen = set()
    for path in Path(directory).glob("*.bin"):
        stem = path.stem
        if not path.is_file() or not stem.isascii() or not stem.isdecimal():
            raise ValueError(f"Sequence input must be a numeric .bin file: {path.name}")
        digits = stem.lstrip("0") or "0"
        if len(digits) > 10 or int(digits) > 0xFFFFFFFF:
            raise ValueError(f"Frame ID is outside the unsigned 32-bit range: {path.name}")
        number = int(digits)
        if number in seen:
            raise ValueError(f"Duplicate frame ID {number}: {path.name}")
        seen.add(number)
        frames.append((number, path))
    if not frames:
        raise ValueError("No input frames")
    return sorted(frames)


def require_new_output(path):
    """Reject files, directories and dangling symlinks without touching them."""
    if os.path.lexists(path):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(path))


def _publish_directory(source, destination):
    """Atomically publish a Linux directory, never replacing an existing path.

    Plain rename/replace can overwrite a directory created by another process
    after our existence check. The supported runtime is Linux; if its libc or
    filesystem lacks RENAME_NOREPLACE, fail rather than use an unsafe fallback.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        rename = libc.renameat2
    except AttributeError:
        raise OSError(
            errno.ENOTSUP,
            "Atomic no-clobber directory publication requires renameat2",
        ) from None
    rename.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


@contextmanager
def staged_output(output, *, directory=False):
    """Publish only a completed output, cleaning only this invocation's staging.

    A unique private sibling directory keeps existing .partial files untouched
    and preserves file extensions for write_points. Publication is no-clobber
    even if another process creates the destination while the codec is running.
    """
    output = Path(output)
    require_new_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.partial-", dir=output.parent
    ) as staging:
        temporary = Path(staging) / output.name
        if directory:
            temporary.mkdir()
        yield temporary
        if directory:
            _publish_directory(temporary, output)
        else:
            # A hard link is atomic and fails if *any* destination entry exists.
            # Staging is a sibling, so source and destination share a filesystem.
            os.link(temporary, output)
