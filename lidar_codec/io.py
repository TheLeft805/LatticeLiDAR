"""XYZ-only file I/O. Reflectance, input order and duplicate multiplicity are not coded."""

from pathlib import Path
import numpy as np


def read_points(path):
    path = Path(path)
    if path.suffix == ".npy":
        points = np.load(path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("NPY input must contain an Nx3 coordinate array")
        return np.ascontiguousarray(points)
    if path.suffix == ".bin":
        if path.stat().st_size % 16:
            raise ValueError(
                "KITTI BIN input must contain four float32 values per point"
            )
        return np.fromfile(path, dtype="<f4").reshape(-1, 4)[:, :3].copy()
    raise ValueError("Input must be a KITTI .bin or an Nx3 .npy file")


def write_points(path, points):
    path = Path(path)
    if path.suffix not in (".npy", ".ply"):
        raise ValueError("Reconstruction must be .npy or .ply")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    points = np.asarray(points, dtype="<f4")
    with path.open("xb") as stream:
        if path.suffix == ".npy":
            np.save(stream, points, allow_pickle=False)
        elif path.suffix == ".ply":
            header = (
                "ply\nformat binary_little_endian 1.0\n"
                f"element vertex {len(points)}\n"
                "property float x\nproperty float y\nproperty float z\nend_header\n"
            )
            stream.write(header.encode("ascii"))
            stream.write(points.tobytes())
        else:
            raise ValueError("Reconstruction must be .npy or .ply")
