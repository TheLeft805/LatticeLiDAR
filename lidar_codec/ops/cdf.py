import torch
from functools import lru_cache
import triton
import triton.language as tl
from triton.language import math as libdevice


@torch.no_grad()
def _convert_cdf16_scalar(cdf_float):
    """Reuse the original device-computed normalization as a Python scalar.

    This opt-in path only covers the maintained 16-class FP32 codec. It keeps
    torch.round and the separate integer addition, without fused arithmetic.
    """
    if cdf_float.dtype != torch.float32 or cdf_float.shape[-1] != 17:
        raise ValueError("Scalar CDF execution requires FP32 rows of length17")
    factor = _cdf16_factor(cdf_float.device)
    cdf = (cdf_float * factor).round().to(dtype=torch.int16, non_blocking=True)
    cdf.add_(torch.arange(17, dtype=torch.int16, device=cdf.device))
    return cdf


@lru_cache(maxsize=32)
def _cdf16_factor(device):
    # CUDA pow(2, 16) need not equal 65536 exactly. Preserve the original
    # kernel's FP32 result; computing the mathematical constant changes CDFs.
    # The one-time device synchronization belongs to codec warm-up.
    return (torch.tensor(2, dtype=torch.float32, device=device).pow_(16) - 16).item()


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@triton.jit(do_not_specialize=["N"])
def _center(
    P,
    XYZ,
    LOOKUP,
    MAP,
    CENTER,
    GROUP,
    N,
    K: tl.constexpr,
    J: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0) * B + tl.arange(0, B)
    axis = tl.arange(0, 4)
    xyz = tl.load(
        XYZ + row[:, None] * 3 + axis[None, :],
        (row[:, None] < N) & (axis[None, :] < 3),
        0,
    )
    previous = tl.load(
        XYZ + (row[:, None] - 1) * 3 + axis[None, :],
        (row[:, None] < N) & (row[:, None] % 32 > 0) & (axis[None, :] < 3),
        0,
    )
    delta = tl.sum(tl.abs(xyz - previous), 1)
    group = (
        (delta > 1).to(tl.int32) + (delta > 4).to(tl.int32) + (delta > 16).to(tl.int32)
    )
    j = tl.arange(0, J)
    c = tl.arange(0, 16)
    weights = tl.load(
        P + (row[:, None] - 1) * 16 + (j[None, :] % 16),
        (row[:, None] < N) & (row[:, None] % 32 > 0),
        0,
    )
    if K == 32:
        weights *= tl.load(
            P + (row[:, None] - 2) * 16 + j[None, :] // 16,
            (row[:, None] < N) & (row[:, None] % 32 > 1),
            0,
        )
        state = tl.load(MAP + j)
    else:
        state = j
    table = tl.load(
        LOOKUP
        + (group[:, None, None] * K + state[None, :, None]) * 16
        + c[None, None, :]
    )
    center = tl.sum(weights[:, :, None] * table, 1)
    tl.store(CENTER + row[:, None] * 16 + c[None, :], center, row[:, None] < N)
    tl.store(GROUP + row, group, row < N)


@triton.jit(do_not_specialize=["N"])
def _head(
    L,
    CENTER,
    GROUP,
    TABLE,
    DELTA,
    Y,
    OUT,
    TRACE_Q,
    BAD,
    LENGTHS,
    OFFSETS,
    N,
    COMPONENTS: tl.constexpr,
    SELECTED: tl.constexpr,
    TRACE: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0) * B + tl.arange(0, B)
    if COMPONENTS == 1:
        # One metadata writer per block, fused with the compulsory base head.
        offset = row * 32
        tl.store(LENGTHS + row, tl.minimum(32, N - offset), offset < N)
        tl.store(OFFSETS + row, offset, offset < N)
    c = tl.arange(0, 16)
    if SELECTED:
        node = row // COMPONENTS
        state = tl.load(Y + node - 1, (node < N) & (node % 32 > 0), 0).to(tl.int32)
    else:
        node = row // (16 * COMPONENTS)
        state = (row // COMPONENTS) % 16
    component = row % COMPONENTS
    base = tl.load(L + node[:, None] * 16 + c[None, :], node[:, None] < N, 0)
    center = tl.load(CENTER + node[:, None] * 16 + c[None, :], node[:, None] < N, 0)
    group = tl.load(GROUP + node, node < N, 0)
    bias = tl.load(TABLE + (group[:, None] * 16 + state[:, None]) * 16 + c[None, :])
    value = base + tl.where((node % 32)[:, None] > 0, bias - center, 0)
    if COMPONENTS == 4:
        value += tl.load(
            DELTA + (node[:, None] * 4 + component[:, None]) * 16 + c[None, :],
            node[:, None] < N,
            0,
        )
    finite = tl.sum((tl.abs(value) < float("inf")).to(tl.int32), 1) == 16
    e = tl.exp(value - tl.max(value, 1)[:, None])
    probability = tl.maximum(e / tl.sum(e, 1)[:, None], 1.0e-30)
    probability = probability / tl.sum(probability, 1)[:, None]
    mass = libdevice.nearbyint(tl.cumsum(probability, 1) * 65520).to(tl.int32)
    mass = tl.associative_scan(tl.minimum(tl.maximum(mass, 0), 65520), 1, _maximum)
    bounds = tl.where(c[None, :] == 15, 65536, mass + c[None, :] + 1)
    if SELECTED:
        y = tl.load(Y + node, node < N, 0).to(tl.int32)
        low = tl.sum(tl.where(c[None, :] == y[:, None] - 1, bounds, 0), 1)
        high = tl.sum(tl.where(c[None, :] == y[:, None], bounds, 0), 1)
        tl.store(OUT + row * 2, low, node < N)
        tl.store(OUT + row * 2 + 1, high - low, node < N)
        if TRACE:
            tl.store(
                TRACE_Q + row[:, None] * 17 + c[None, :] + 1, bounds, node[:, None] < N
            )
            tl.store(TRACE_Q + row * 17, 0, node < N)
    else:
        tl.store(OUT + row[:, None] * 17 + c[None, :] + 1, bounds, node[:, None] < N)
        tl.store(OUT + row * 17, 0, node < N)
    if tl.sum(((node < N) & ~finite).to(tl.int32), 0) > 0:
        tl.atomic_max(BAD, 1)


@triton.jit(do_not_specialize=["N"])
def _coarse(Q, Y, OUT, BAD, LENGTHS, OFFSETS, N, SELECTED: tl.constexpr, B: tl.constexpr):
    node = tl.program_id(0) * B + tl.arange(0, B)
    offset = node * 32
    tl.store(LENGTHS + node, tl.minimum(32, N - offset), offset < N)
    tl.store(OFFSETS + node, offset, offset < N)
    if SELECTED:
        y = tl.load(Y + node, node < N, 0).to(tl.int32)
        low = tl.load(Q + node * 17 + y, node < N, 0).to(tl.int32) & 65535
        high = tl.load(Q + node * 17 + y + 1, node < N, 0).to(tl.int32) & 65535
        high = tl.where(y == 15, 65536, high)
        tl.store(OUT + node * 2, low, node < N)
        tl.store(OUT + node * 2 + 1, high - low, node < N)
        if tl.sum(((node < N) & (high <= low)).to(tl.int32), 0) > 0:
            tl.atomic_max(BAD, 1)
    else:
        c = tl.arange(0, 32)
        value = (
            tl.load(
                Q + node[:, None] * 17 + c[None, :],
                (node[:, None] < N) & (c[None, :] < 17),
                0,
            ).to(tl.int32)
            & 65535
        )
        value = tl.where(c[None, :] == 16, 65536, value)
        tl.store(
            OUT + node[:, None] * 17 + c[None, :],
            value,
            (node[:, None] < N) & (c[None, :] < 17),
        )


@triton.jit(do_not_specialize=["N"])
def _block_metadata(LENGTHS, OFFSETS, N, B: tl.constexpr):
    block = tl.program_id(0) * B + tl.arange(0, B)
    offset = block * 32
    tl.store(LENGTHS + block, tl.minimum(32, N - offset), offset < N)
    tl.store(OFFSETS + block, offset, offset < N)


def prepare(pred, coordinates, scale, table, bad, target=None, trace=False):
    n = len(pred.rows)
    selected = target is not None
    blocks = (n + 31) // 32
    device = pred.logits.device
    lengths = torch.empty(blocks, device=device, dtype=torch.int32)
    offsets = torch.empty_like(lengths)
    d = dict(
        lengths=lengths,
        offsets=offsets,
        dynamic=int(scale <= 3),
        k4=int(pred.delta is not None),
        dense_cdf16=not selected and scale <= 3,
        n=n,
        priors=(
            pred.prior
            if pred.prior is not None
            # Native non-K4 paths never dereference the prior pointer.
            else bad
        ),
    )
    dummy = pred.logits
    y = target if selected else dummy

    def output(components):
        return torch.empty(
            (n, components, 2) if selected else (n, 16, components, 17),
            device=device,
            dtype=torch.int32 if selected else torch.int16,
        )

    if scale > 3:
        d["base"] = torch.empty(
            (n, 2 if selected else 17), device=device, dtype=torch.int32
        )
        _coarse[(triton.cdiv(n, 128),)](
            pred.cdf, y, d["base"], bad, lengths, offsets, n, selected, 128, num_warps=4
        )
        d["components"] = d["base"]
        if trace:
            d["trace_base"] = pred.cdf.int() & 65535
            d["trace_base"][:, -1] = 65536
    else:
        center = torch.empty((n, 16), device=device)
        group = torch.empty(n, device=device, dtype=torch.int32)
        _center[(triton.cdiv(n, 32),)](
            pred.logits.softmax(-1),
            coordinates.contiguous(),
            table.lookup,
            table.assignments,
            center,
            group,
            n,
            16,
            16,
            32,
            num_warps=4,
            enable_fp_fusion=False,
        )
        for key, components in [("base", 1)] + ([("components", 4)] if d["k4"] else []):
            q = output(components)
            qt = (
                torch.empty((n, components, 17), device=device, dtype=torch.int32)
                if trace and selected
                else dummy
            )
            _head[(triton.cdiv(n * components * (1 if selected else 16), 128),)](
                pred.logits,
                center,
                group,
                table.lookup,
                pred.delta if d["k4"] else dummy,
                y,
                q,
                qt,
                bad,
                lengths,
                offsets,
                n,
                components,
                selected,
                trace,
                128,
                num_warps=4,
                enable_fp_fusion=False,
            )
            d[key] = q
            if trace and selected:
                d["trace_" + key] = qt
        if not d["k4"]:
            d["components"] = d["base"]
    if selected:
        d.update(
            # prepare_group overwrites every K4 block before encode_group reads
            # it. Non-K4 encoding never dereferences either pointer.
            paid=torch.empty((blocks, 4), device=device, dtype=torch.int32) if d["k4"] else bad,
            post=torch.empty((blocks, 5), device=device, dtype=torch.int32) if d["k4"] else bad,
        )
    else:
        d.update(
            output=torch.zeros(n, device=device, dtype=torch.int16),
            posterior=torch.empty((blocks, 4), device=device, dtype=torch.int64) if d["k4"] else bad,
        )
    return d
