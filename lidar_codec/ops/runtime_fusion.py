"""FP32 arithmetic at the original FP16 rounding boundaries, and owned gathers."""
import torch
import triton
import triton.language as tl
from triton.language import math as libdevice


@triton.jit
def _silu_half(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    x = tl.load(X+i, i < N, 0).to(tl.float32)
    y = libdevice.div_rn(x, 1.0+libdevice.exp(-x))
    tl.store(Y+i, y, i < N)


def silu_half(value):
    output = torch.empty_like(value)
    _silu_half[(triton.cdiv(value.numel(), 256),)](
        value, output, value.numel(), 256, enable_fp_fusion=False)
    return output


@triton.jit
def _static(Rows, Sensor, History, HistoryMask, Beam, Sub, Near, Far, Offsets, Has,
            OwnOut, SensorOut, HistoryOut, HistoryMaskOut, BeamOut, TokenOut,
            MaskOut, OffsetsOut, AvailableOut, GROUP: tl.constexpr, FINE: tl.constexpr,
            ROWS: tl.constexpr):
    row = tl.program_id(0)*ROWS+tl.arange(0, ROWS)[:, None]
    source = tl.load(Rows+row)
    j = tl.arange(0, 32)[None, :]
    tl.store(OwnOut+row, 16)
    tl.store(SensorOut+row*5+j, tl.load(Sensor+source*5+j, j < 5, 0), j < 5)
    tl.store(HistoryOut+row*27+j, tl.load(History+source*27+j, j < 27, 0), j < 27)
    tl.store(HistoryMaskOut+row*27+j, tl.load(HistoryMask+source*27+j, j < 27, 0), j < 27)
    tl.store(BeamOut+row*21+j, tl.load(Beam+source*21+j, j < 21, 0), j < 21)
    tl.store(TokenOut+row*30+j, 16, j < 30)
    if GROUP == 0:
        mask = tl.full((ROWS, 32), False, tl.int1)
    elif GROUP == 1:
        mask = tl.load(Sub+row*8+j-22, (j >= 22) & (j < 30), 0)
    else:
        mask = tl.load(Near+row*14+j, j < 14, 0) | tl.load(Far+row*8+j-14, (j >= 14) & (j < 22), 0)
    tl.store(MaskOut+row*30+j, mask, j < 30)
    if GROUP == 2:
        offsets = tl.load(Offsets+row*24+j, j < 24, 0)
    else:
        offsets = tl.full((ROWS, 32), 0, tl.int8)
    tl.store(OffsetsOut+row*24+j, offsets, j < 24)
    # Explicit scalar columns avoid the incorrect bool broadcast observed with
    # the 8-row layout on the frozen Triton runtime. All bytes are canonical 0/1.
    tl.store(AvailableOut+row*4, 1 if GROUP == 2 else 0)
    tl.store(AvailableOut+row*4+1, 1 if GROUP == 1 else 0)
    tl.store(AvailableOut+row*4+2, (tl.load(Has) != 0).to(tl.uint8))
    tl.store(AvailableOut+row*4+3, 1 if FINE else 0)


def static_outputs(workspace, capacity, rows, sensor, history, history_mask,
                   beam, sub, near, far, offsets, has_history, scale):
    """Every output element is overwritten, including padded rows after poison."""
    shapes = dict(own0=((), torch.long), sensor=((5,), torch.float32),
                  history=((27,), torch.uint8), history_mask=((27,), torch.bool),
                  beam=((21,), torch.float32), tokens=((30,), torch.uint8),
                  mask=((30,), torch.bool), offsets=((8, 3), torch.int8),
                  available=((4,), torch.bool))
    outputs = []
    for group, indices in enumerate(rows):
        out = {name: workspace.view((capacity, *tail), dtype, sensor.device,
                                   capacity, ('static', group, name))
               for name, (tail, dtype) in shapes.items()}
        # Eight rows per CTA fills the 128 threads instead of one mostly idle warp.
        _static[(capacity//8,)](indices, sensor, history, history_mask, beam, sub,
            near, far, offsets, has_history, *out.values(), group, scale <= 3, 8,
            num_warps=4, enable_fp_fusion=False)
        outputs.append(out)
    return outputs


@triton.jit
def _role_rows(M0, M1, M2, M3, R0, R1, R2, R3, O0, O1, O2, O3,
               N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    active = row < N
    # Active ranks occupy [0,count); padding writes occupy [count,N).
    # These destinations are disjoint even across CTAs, so no atomics/barrier.
    count0 = tl.load(R0+N-1)+1
    count1 = tl.load(R1+N-1)+1
    count2 = tl.load(R2+N-1)+1
    count3 = tl.load(R3+N-1)+1
    tl.store(O0+row, N-1, active & (row >= count0))
    tl.store(O1+row, N-1, active & (row >= count1))
    tl.store(O2+row, N-1, active & (row >= count2))
    tl.store(O3+row, N-1, active & (row >= count3))
    tl.store(O0+tl.load(R0+row, active, 0), row, active & tl.load(M0+row, active, False).to(tl.int1))
    tl.store(O1+tl.load(R1+row, active, 0), row, active & tl.load(M1+row, active, False).to(tl.int1))
    tl.store(O2+tl.load(R2+row, active, 0), row, active & tl.load(M2+row, active, False).to(tl.int1))
    tl.store(O3+tl.load(R3+row, active, 0), row, active & tl.load(M3+row, active, False).to(tl.int1))


def stable_role_rows(workspace, capacity, masks):
    """Stable integer compaction equals sort(where(mask, arange, capacity)).clamp."""
    ranks = [mask.long().cumsum(0)-1 for mask in masks]
    roles = [workspace.view((capacity,), torch.long, masks[0].device, capacity, ('roles', i))
             for i in range(4)]
    _role_rows[(triton.cdiv(capacity, 256),)](*masks, *ranks, *roles, capacity, 256)
    return roles, ranks
