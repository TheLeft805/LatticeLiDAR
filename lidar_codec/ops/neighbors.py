"""Ordered, integer-exact causal neighborhoods."""

from itertools import product
from functools import lru_cache
import torch
import triton
import triton.language as tl

OFFSETS = tuple(product((-1, 0, 1), repeat=3))
NEAR_OFFSETS = tuple(d for d in OFFSETS if sum(d) % 2 == 1)
FAR_CANDIDATES = tuple(
    sorted(
        (
            d
            for d in product(range(-3, 4), repeat=3)
            if 1 < max(map(abs, d)) <= 3 and sum(d) % 2 == 1
        ),
        key=lambda d: (sum(v * v for v in d), d),
    )
)
SUB = tuple(d for d in OFFSETS if abs(d[0]) == 1 and abs(d[1]) + abs(d[2]) == 1)
HISTORY = OFFSETS


def packed_keys(xyz):
    """Injective int64 key for xyz in [0, 2**21)^3."""
    xyz = xyz.to(torch.int64)
    return (xyz[..., 0] << 42) | (xyz[..., 1] << 21) | xyz[..., 2]


@triton.jit
def _dilate3(v):
    v = v.to(tl.int64) & 0x1FFFFF
    v = (v | (v << 32)) & 0x1F00000000FFFF
    v = (v | (v << 16)) & 0x1F0000FF0000FF
    v = (v | (v << 8)) & 0x100F00F00F00F00F
    v = (v | (v << 4)) & 0x10C30C30C30C30C3
    v = (v | (v << 2)) & 0x1249249249249249
    return v


@triton.jit(do_not_specialize=["N", "STEPS"])
def _offset_rows(
    X,
    KEYS,
    ORDER,
    DELTA,
    ROWS,
    MASK,
    N,
    S0: tl.constexpr,
    S1: tl.constexpr,
    STEPS,
    SLOTS: tl.constexpr,
    MORTON: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.arange(0, BLOCK)
    active = slot < SLOTS
    x = tl.load(X + row * S0).to(tl.int64) + tl.load(DELTA + slot * 3, active, 0).to(
        tl.int64
    )
    y = tl.load(X + row * S0 + S1).to(tl.int64) + tl.load(
        DELTA + slot * 3 + 1, active, 0
    ).to(tl.int64)
    z = tl.load(X + row * S0 + 2 * S1).to(tl.int64) + tl.load(
        DELTA + slot * 3 + 2, active, 0
    ).to(tl.int64)
    valid = (
        active
        & (x >= 0)
        & (x < 2097152)
        & (y >= 0)
        & (y < 2097152)
        & (z >= 0)
        & (z < 2097152)
    )
    x = tl.where(valid, x, 0)
    y = tl.where(valid, y, 0)
    z = tl.where(valid, z, 0)
    if MORTON:
        key = _dilate3(x) | (_dilate3(y) << 1) | (_dilate3(z) << 2)
    else:
        key = (x << 42) | (y << 21) | z
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), 0, tl.int32) + N
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        value = tl.load(KEYS + tl.minimum(mid, N - 1))
        search = lo < hi
        advance = search & (mid < N) & (value < key)
        lo = tl.where(advance, mid + 1, lo)
        hi = tl.where(search & ~advance, mid, hi)
    pos = tl.minimum(lo, N - 1)
    match = valid & (lo < N) & (tl.load(KEYS + pos) == key)
    value = tl.where(match, tl.load(ORDER + pos), 0)
    tl.store(ROWS + row * SLOTS + slot, value, active)
    tl.store(MASK + row * SLOTS + slot, match, active)


@lru_cache(maxsize=16)
def offsets(device, values):
    return torch.tensor(values, dtype=torch.int32, device=device)


def lookup(xyz, keys, order, delta, *, morton=False):
    shape = (len(xyz), len(delta))
    rows = torch.zeros(shape, dtype=torch.long, device=xyz.device)
    mask = torch.zeros(shape, dtype=torch.bool, device=xyz.device)
    if len(xyz) and len(keys):
        d = offsets(xyz.device, tuple(delta))
        _offset_rows[(len(xyz),)](
            xyz,
            keys,
            order,
            d,
            rows,
            mask,
            len(keys),
            xyz.stride(0),
            xyz.stride(1),
            len(keys).bit_length(),
            len(delta),
            morton,
            triton.next_power_of_2(len(delta)),
            num_warps=4,
        )
    return rows, mask


@lru_cache(maxsize=8)
def _device_offsets(device_index, offsets):
    return torch.tensor(
        offsets, dtype=torch.int32, device=torch.device("cuda", device_index)
    )


@triton.jit(do_not_specialize=["NA", "STEPS"])
def _fused_rows(
    B,
    KEYS,
    ORDER,
    OFFSETS,
    NEAR_ROWS,
    NEAR_MASK,
    FAR_ROWS,
    FAR_MASK,
    FAR_OFFSETS,
    NA,
    SB0: tl.constexpr,
    SB1: tl.constexpr,
    STEPS,
    LIMIT: tl.constexpr,
    NNEAR: tl.constexpr,
    NFAR: tl.constexpr,
    NCHOSEN: tl.constexpr,
    BSIZE: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.arange(0, BSIZE)
    active = slot < NNEAR + NFAR
    dx = tl.load(OFFSETS + slot * 3 + 0, active, other=0).to(tl.int64)
    dy = tl.load(OFFSETS + slot * 3 + 1, active, other=0).to(tl.int64)
    dz = tl.load(OFFSETS + slot * 3 + 2, active, other=0).to(tl.int64)
    x = tl.load(B + row * SB0 + 0 * SB1).to(tl.int64) + dx
    y = tl.load(B + row * SB0 + 1 * SB1).to(tl.int64) + dy
    z = tl.load(B + row * SB0 + 2 * SB1).to(tl.int64) + dz
    valid = (
        active
        & (x >= 0)
        & (x < LIMIT)
        & (y >= 0)
        & (y < LIMIT)
        & (z >= 0)
        & (z < LIMIT)
    )
    # Bounds precede key construction: invalid coordinates cannot wrap into
    # a valid signed-63-bit Morton-domain key or match key zero.
    x = tl.where(valid, x, 0)
    y = tl.where(valid, y, 0)
    z = tl.where(valid, z, 0)
    query = (x << 42) | (y << 21) | z
    low = tl.full((BSIZE,), 0, tl.int32)
    high = tl.full((BSIZE,), 0, tl.int32) + NA
    # Each live interval at least halves; NA.bit_length() iterations
    # suffice. The searching guard keeps converged lanes fixed while
    # other candidates continue their binary search.
    for _ in range(STEPS):
        mid = (low + high) // 2
        current = tl.load(KEYS + tl.minimum(mid, NA - 1))
        searching = low < high
        advance = searching & (mid < NA) & (current < query)
        low = tl.where(advance, mid + 1, low)
        high = tl.where(searching & ~advance, mid, high)
    pos = tl.minimum(low, NA - 1)
    matched = valid & (low < NA) & (tl.load(KEYS + pos) == query)
    source = tl.where(matched, tl.load(ORDER + pos), 0)
    near = slot < NNEAR
    tl.store(NEAR_ROWS + row * NNEAR + slot, source, near)
    tl.store(NEAR_MASK + row * NNEAR + slot, matched, near)
    far_hit = matched & (slot >= NNEAR)
    rank = tl.cumsum(far_hit.to(tl.int32), 0) - 1
    chosen = far_hit & (rank < NCHOSEN)
    tl.store(FAR_ROWS + row * NCHOSEN + rank, source, chosen)
    tl.store(FAR_MASK + row * NCHOSEN + rank, True, chosen)
    tl.store(FAR_OFFSETS + row * NCHOSEN * 3 + rank * 3 + 0, dx.to(tl.int8), chosen)
    tl.store(FAR_OFFSETS + row * NCHOSEN * 3 + rank * 3 + 1, dy.to(tl.int8), chosen)
    tl.store(FAR_OFFSETS + row * NCHOSEN * 3 + rank * 3 + 2, dz.to(tl.int8), chosen)


@triton.jit
def _history(
    X,
    KEYS,
    ORDER,
    OCC,
    ENABLED,
    DELTA,
    H,
    M,
    N: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.arange(0, BLOCK)
    active = slot < 27
    if tl.load(ENABLED):
        x = tl.load(X + row * S0).to(tl.int64) + tl.load(
            DELTA + slot * 3, active, 0
        ).to(tl.int64)
        y = tl.load(X + row * S0 + S1).to(tl.int64) + tl.load(
            DELTA + slot * 3 + 1, active, 0
        ).to(tl.int64)
        z = tl.load(X + row * S0 + 2 * S1).to(tl.int64) + tl.load(
            DELTA + slot * 3 + 2, active, 0
        ).to(tl.int64)
        valid = (
            active
            & (x >= 0)
            & (x < 2097152)
            & (y >= 0)
            & (y < 2097152)
            & (z >= 0)
            & (z < 2097152)
        )
        x = tl.where(valid, x, 0)
        y = tl.where(valid, y, 0)
        z = tl.where(valid, z, 0)
        key = _dilate3(x) | (_dilate3(y) << 1) | (_dilate3(z) << 2)
        lo = tl.full((BLOCK,), 0, tl.int32)
        hi = tl.full((BLOCK,), N, tl.int32)
        for _ in range(STEPS):
            mid = (lo + hi) // 2
            value = tl.load(KEYS + tl.minimum(mid, N - 1))
            search = lo < hi
            advance = search & (mid < N) & (value < key)
            lo = tl.where(advance, mid + 1, lo)
            hi = tl.where(search & ~advance, mid, hi)
        pos = tl.minimum(lo, N - 1)
        match = valid & (lo < N) & (tl.load(KEYS + pos) == key)
        source = tl.load(ORDER + pos)
        value = tl.load(OCC + source, match, 0)
    else:
        match = tl.full((BLOCK,), False, tl.int1)
        value = tl.full((BLOCK,), 0, tl.uint8)
    tl.store(H + row * 27 + slot, value, active)
    tl.store(M + row * 27 + slot, match, active)


def history_values(xyz, keys, order, occupancy, enabled):
    if len(keys) == 0 or len(OFFSETS) != 27:
        raise ValueError("Expected padded reference and 27 slots")
    h = torch.empty((len(xyz), 27), device=xyz.device, dtype=torch.uint8)
    m = torch.empty_like(h, dtype=torch.bool)
    delta = offsets(xyz.device, tuple(OFFSETS))
    _history[(len(xyz),)](
        xyz,
        keys,
        order,
        occupancy,
        enabled,
        delta,
        h,
        m,
        len(keys),
        xyz.stride(0),
        xyz.stride(1),
        len(keys).bit_length(),
        32,
        num_warps=4,
    )
    return h, m
