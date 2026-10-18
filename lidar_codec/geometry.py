from dataclasses import dataclass
import math
import torch

"""One quantization convention for training, encoding, reconstruction and registration."""

import numpy as np

ORIGIN_MM = 131072


def quantize(points, config):
    points = np.asarray(points)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError("Expected a nonempty finite Nx3 point cloud")
    if points.dtype.kind not in "fiu":
        raise ValueError("Coordinates must be real numeric values")
    # Match the original codec: BIN float32 arithmetic stays float32, including ties.
    values = points.copy() if points.dtype.kind == "f" else points.astype(np.int64)
    if config.input_units == "meters":
        values = values / 0.001 + ORIGIN_MM
    elif config.input_units == "millimeters":
        values = values + ORIGIN_MM
    elif config.input_units == "legacy_prequantized" and values.min() < 0:
        values = values + ORIGIN_MM
    quantized = np.rint(values / config.posQ)
    if np.any(quantized < 0) or np.any(quantized >= 2**21):
        raise ValueError(
            "Quantized geometry exceeds the nonnegative 21-bit Morton domain"
        )
    return quantized.astype(np.int32)


def restore(quantized, config):
    millimeters = np.asarray(quantized, dtype=np.int64) * config.posQ
    if config.input_units == "offset_mm":
        return millimeters
    millimeters = millimeters - ORIGIN_MM
    if config.input_units == "meters":
        return (millimeters * 0.001).astype(np.float32)
    return millimeters


def metric_coordinates(quantized, posQ):
    return (np.asarray(quantized, dtype=np.float64) * posQ - ORIGIN_MM) * 0.001


@dataclass(frozen=True)
class Quantization:
    posQ: int = 16
    input_units: str = "meters"

    def __post_init__(self):
        if type(self.posQ) is not int or not 1 <= self.posQ <= 32767:
            raise ValueError(
                "Quantization must be an integer number of millimeters in 1..32767"
            )


def sensor_conditions(xyz, scale, posQ):
    """Representative center of descendant quantized indices, not raw-point centroid.

    A parent coordinate c covers integer indices [c*2**scale, (c+1)*2**scale-1].
    Its midpoint is mapped to meters using the codec's fixed offset convention.
    The unit vector avoids angular discontinuities and undefined atan2 at origin.
    """
    if type(scale) is not int or not 1 <= scale <= 21:
        raise ValueError("Expected 1..21 remaining octree levels")
    if type(posQ) is not int or not 1 <= posQ <= 32767:
        raise ValueError("S1 conditions require explicit valid stream quantization")
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or xyz.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Expected integer coarse Nx3 coordinates")
    side = 2**scale
    center = (xyz.to(torch.float32) * side + (side - 1) * 0.5) * posQ
    meters = (center - ORIGIN_MM) * 0.001
    radius = torch.linalg.vector_norm(meters, dim=1, keepdim=True)
    direction = meters / radius.clamp_min(1.0e-6)
    distance = torch.log1p(radius) / math.log(257.0)
    physical_scale = torch.full_like(radius, math.log2(side * posQ) / 32.0)
    return torch.cat((distance, direction, physical_scale), dim=1)
