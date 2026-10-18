"""Copy occupied child coordinates/features in one kernel, preserving Morton order."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N'])
def _gather(C, F, I, OC, OF, N, CS:tl.constexpr, FS:tl.constexpr,
            FSTRIDE:tl.constexpr, FEATURES:tl.constexpr, B:tl.constexpr):
    row = tl.program_id(0)*B + tl.arange(0, B)
    selected = tl.load(I+row, row<N, 0).to(tl.int32)
    parent = selected >> 3
    child = selected & 7
    # Per-column expressions avoid mixed 4/32-column broadcast lowering in
    # Triton 2.2 (the eight-child regression checks each axis independently).
    batch = tl.load(C+parent*CS, row<N, 0)*2
    x = tl.load(C+parent*CS+1, row<N, 0)*2 + (child & 1)
    y = tl.load(C+parent*CS+2, row<N, 0)*2 + ((child >> 1) & 1)
    z = tl.load(C+parent*CS+3, row<N, 0)*2 + ((child >> 2) & 1)
    tl.store(OC+row*4, batch, row<N)
    tl.store(OC+row*4+1, x, row<N)
    tl.store(OC+row*4+2, y, row<N)
    tl.store(OC+row*4+3, z, row<N)
    if FEATURES:
        channel = tl.arange(0, 32)
        value = tl.load(F+parent[:,None]*FS+channel[None,:]*FSTRIDE, row[:,None]<N, 0)
        tl.store(OF+row[:,None]*32+channel[None,:], value, row[:,None]<N)


def expand(coordinates, occupancy, features=None):
    if len(coordinates) >= 2**28 or coordinates.stride(1) != 1 or (features is not None and features.shape[1] != 32):
        raise ValueError('Expected four contiguous coordinate columns and 32 feature channels')
    bits = torch.arange(8, device=occupancy.device)
    keep = ((occupancy.reshape(-1, 1).long() >> bits) & 1).bool()
    selected = keep.reshape(-1).nonzero(as_tuple=False).flatten()
    n = len(selected)
    output = torch.empty((n, 4), dtype=torch.int32, device=coordinates.device)
    propagated = None if features is None else features.new_empty((n, 32))
    if n:
        _gather[(triton.cdiv(n, 128),)](coordinates, features if features is not None else coordinates,
            selected, output, propagated if propagated is not None else output, n,
            coordinates.stride(0), 0 if features is None else features.stride(0),
            0 if features is None else features.stride(1), features is not None, 128, num_warps=4)
    return output, propagated


@torch.no_grad()
def upscale_coordinate(coordinates, occupancy):
    return expand(coordinates, occupancy)[0]


@torch.no_grad()
def upscale_coordinate_feature(coordinates, occupancy, features):
    return expand(coordinates, occupancy, features)
