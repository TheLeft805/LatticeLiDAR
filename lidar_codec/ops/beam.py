"""Full 64-beam fused features from decoder-owned integer parent coordinates."""

import torch
import triton
import triton.language as tl
import triton.language.math as libdevice


@triton.jit(do_not_specialize=["N"])
def beam_kernel(
    Q,
    Table,
    Normal,
    Out,
    Nearest,
    Crossings,
    Refined,
    N,
    RS: tl.constexpr,
    CS: tl.constexpr,
    SIDE: tl.constexpr,
    QSTEP: tl.constexpr,
    DIAG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // 8
    child = i % 8
    valid = row < N
    qx = tl.load(Q + row * RS, valid, 0).to(tl.int64)
    qy = tl.load(Q + row * RS + CS, valid, 0).to(tl.int64)
    qz = tl.load(Q + row * RS + 2 * CS, valid, 0).to(tl.int64)
    xi = (2 * qx * SIDE + (child & 1) * SIDE + SIDE // 2 - 1) * QSTEP - 262144
    yi = (2 * qy * SIDE + ((child >> 1) & 1) * SIDE + SIDE // 2 - 1) * QSTEP - 262144
    zi = (2 * qz * SIDE + ((child >> 2) & 1) * SIDE + SIDE // 2 - 1) * QSTEP - 262144
    x = xi.to(tl.float32) / 2000.0
    y = yi.to(tl.float32) / 2000.0
    z = zi.to(tl.float32) / 2000.0
    width = SIDE * QSTEP * 0.0005
    half = width / 2
    j = tl.arange(0, 64)
    a = tl.load(Table + 2 * j).to(tl.float32)
    b = tl.load(Table + 2 * j + 1).to(tl.float32)
    normal = tl.load(Normal + j).to(tl.float32)
    rho = tl.sqrt(x * x + y * y)
    e = (z[:, None] - rho[:, None] * a[None, :] - b[None, :]) / normal[None, :]
    magnitude = tl.abs(e)
    best = tl.min(magnitude, 1)
    idx = tl.min(tl.where(magnitude == best[:, None], j[None, :], 64), 1)
    second = tl.min(tl.where(j[None, :] != idx[:, None], magnitude, float("inf")), 1)
    lo_x = tl.maximum(tl.abs(x) - half, 0.0)
    lo_y = tl.maximum(tl.abs(y) - half, 0.0)
    hi_x = tl.abs(x) + half
    hi_y = tl.abs(y) + half
    rlo = tl.sqrt(lo_x * lo_x + lo_y * lo_y)
    rhi = tl.sqrt(hi_x * hi_x + hi_y * hi_y)
    elo = (
        z[:, None]
        - half
        - b[None, :]
        - tl.maximum(rlo[:, None] * a[None, :], rhi[:, None] * a[None, :])
    )
    ehi = (
        z[:, None]
        + half
        - b[None, :]
        - tl.minimum(rlo[:, None] * a[None, :], rhi[:, None] * a[None, :])
    )
    count = tl.sum(((elo <= 0) & (ehi >= 0)).to(tl.int32), 1)
    # Conservative near-decision band; all 64 beams are reevaluated in FP64.
    eps = 1.0e-5 * (tl.abs(z) + rho + 1.0)
    near = (second - best <= eps) | (
        tl.min(tl.minimum(tl.abs(elo), tl.abs(ehi)), 1) <= eps
    )
    # FP64 refinement repeats the reference coordinate arithmetic, including
    # its final offset subtraction, to preserve exact boundary signs.
    unit_d = tl.full((), QSTEP * 0.0005, tl.float64)
    origin_d = tl.full((), 131.072, tl.float64)
    width_d = tl.full((), SIDE * QSTEP * 0.0005, tl.float64)
    half_d = width_d / 2.0
    xd = (2 * qx * SIDE + (child & 1) * SIDE + SIDE // 2 - 1).to(
        tl.float64
    ) * unit_d - origin_d
    yd = (2 * qy * SIDE + ((child >> 1) & 1) * SIDE + SIDE // 2 - 1).to(
        tl.float64
    ) * unit_d - origin_d
    zd = (2 * qz * SIDE + ((child >> 2) & 1) * SIDE + SIDE // 2 - 1).to(
        tl.float64
    ) * unit_d - origin_d
    rhod = libdevice.sqrt_rn(xd * xd + yd * yd)
    refine = tl.sum((near & valid).to(tl.int32), 0) > 0
    if refine:
        ad = tl.load(Table + 2 * j)
        bd = tl.load(Table + 2 * j + 1)
        nd = tl.load(Normal + j)
        ed = (zd[:, None] - rhod[:, None] * ad[None, :] - bd[None, :]) / nd[None, :]
        bestd = tl.min(tl.abs(ed), 1)
        idx = tl.min(tl.where(tl.abs(ed) == bestd[:, None], j[None, :], 64), 1)
        ld_x = tl.maximum(tl.abs(xd) - half_d, 0.0)
        ld_y = tl.maximum(tl.abs(yd) - half_d, 0.0)
        hd_x = tl.abs(xd) + half_d
        hd_y = tl.abs(yd) + half_d
        rld = libdevice.sqrt_rn(ld_x * ld_x + ld_y * ld_y)
        rhd = libdevice.sqrt_rn(hd_x * hd_x + hd_y * hd_y)
        eld = (
            zd[:, None]
            - half_d
            - bd[None, :]
            - tl.maximum(rld[:, None] * ad[None, :], rhd[:, None] * ad[None, :])
        )
        ehd = (
            zd[:, None]
            + half_d
            - bd[None, :]
            - tl.minimum(rld[:, None] * ad[None, :], rhd[:, None] * ad[None, :])
        )
        count = tl.sum(((eld <= 0) & (ehd >= 0)).to(tl.int32), 1)
    # Only the selected residual uses FP64 on the common path, preventing
    # amplification of coordinate rounding by the narrow child-cell width.
    selected = (
        zd - rhod * tl.load(Table + 2 * idx) - tl.load(Table + 2 * idx + 1)
    ) / tl.load(Normal + idx)
    residual = tl.minimum(tl.maximum(selected / width_d, -8.0), 8.0).to(tl.float32)
    tl.store(Out + row * 21 + 5 + 2 * child, residual, valid)
    tl.store(
        Out + row * 21 + 6 + 2 * child, tl.minimum(count, 4).to(tl.float32) / 4.0, valid
    )
    cx = ((2 * qx * SIDE + SIDE - 1) * QSTEP - 262144).to(tl.float32) / 2000.0
    cy = ((2 * qy * SIDE + SIDE - 1) * QSTEP - 262144).to(tl.float32) / 2000.0
    cz = ((2 * qz * SIDE + SIDE - 1) * QSTEP - 262144).to(tl.float32) / 2000.0
    radius = tl.sqrt(cx * cx + cy * cy + cz * cz)
    denom = tl.maximum(radius, 1.0e-6)
    first = valid & (child == 0)
    tl.store(Out + row * 21, tl.log(1.0 + radius) / 5.54907608489522, first)
    tl.store(Out + row * 21 + 1, cx / denom, first)
    tl.store(Out + row * 21 + 2, cy / denom, first)
    tl.store(Out + row * 21 + 3, cz / denom, first)
    tl.store(Out + row * 21 + 4, libdevice.log2(SIDE * QSTEP * 1.0) / 32.0, first)
    if DIAG:
        tl.store(Nearest + i, idx, valid)
        tl.store(Crossings + i, count, valid)
        tl.store(Refined + i, refine, valid)


class BeamFeatures:
    """Immutable-table inference provider; reference provider remains intact."""

    def __init__(self, reference):
        self.enabled = True
        self.table = reference.table
        self.normal = reference.normal
        if (
            self.table.shape != (64, 2)
            or self.table.dtype != torch.float64
            or not self.table.is_cuda
        ):
            raise ValueError("Expected existing authorized CUDA FP64 64-beam table")
        self.version = (
            id(self.table),
            self.table._version,
            id(self.normal),
            self.normal._version,
        )
        self.calls = 0
        self.parents = 0

    def __call__(self, coords, scale, posQ=16, *, diagnostic=False):
        if (
            coords.ndim != 2
            or coords.shape[1] != 3
            or coords.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Expected integer parent XYZ")
        if (
            not 1 <= scale <= 3
            or not 1 <= posQ <= 32767
            or coords.device != self.table.device
        ):
            raise ValueError(
                "Frozen beam profile requires scales 1-3, valid millimeter step, same device"
            )
        if (
            torch.is_grad_enabled()
            or (
                id(self.table),
                self.table._version,
                id(self.normal),
                self.normal._version,
            )
            != self.version
        ):
            raise RuntimeError("Frozen beam provider invalidated")
        self.calls += 1
        self.parents += len(coords)
        out = torch.empty((len(coords), 21), device=coords.device, dtype=torch.float32)
        diag = (
            [
                torch.empty((len(coords), 8), device=coords.device, dtype=torch.int32)
                for _ in range(3)
            ]
            if diagnostic
            else [out] * 3
        )
        if len(coords):
            beam_kernel[(triton.cdiv(len(coords) * 8, 16),)](
                coords,
                self.table,
                self.normal,
                out,
                *diag,
                len(coords),
                coords.stride(0),
                coords.stride(1),
                2**scale,
                posQ,
                diagnostic,
                16,
                num_warps=4,
                enable_fp_fusion=False
            )
        if not self.enabled:
            out[:, 5:].zero_()
        return (out, *diag) if diagnostic else out

    def stats(self):
        return dict(
            version="full64-stable-fp32-boundary-fp64-v2",
            calls=self.calls,
            parents=self.parents,
            table_bytes=self.table.numel() * self.table.element_size()
            + self.normal.numel() * self.normal.element_size(),
            persistent_workspace_bytes=0,
            beam_candidates=64,
        )
