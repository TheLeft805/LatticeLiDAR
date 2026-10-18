"""Encoder-only delayed coarse descriptors; all probability values stay unchanged."""
import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_batch(META, BASE, LENGTHS, OFFSETS, BAD, B: tl.constexpr):
    group = tl.program_id(0)
    row = tl.program_id(1)*B + tl.arange(0, B)
    q = tl.load(META+group*5).to(tl.pointer_type(tl.int16))
    y = tl.load(META+group*5+1).to(tl.pointer_type(tl.int32))
    n = tl.load(META+group*5+2).to(tl.int32)
    base_offset = tl.load(META+group*5+3).to(tl.int32)
    block_offset = tl.load(META+group*5+4).to(tl.int32)
    label = tl.load(y+row, row<n, 0).to(tl.int32)
    low = tl.load(q+row*17+label, row<n, 0).to(tl.int32)&65535
    high = tl.load(q+row*17+label+1, row<n, 0).to(tl.int32)&65535
    high = tl.where(label==15, 65536, high)
    tl.store(BASE+(base_offset+row)*2, low, row<n)
    tl.store(BASE+(base_offset+row)*2+1, high-low, row<n)
    offset = row*32
    tl.store(LENGTHS+block_offset+row, tl.minimum(32, n-offset), offset<n)
    tl.store(OFFSETS+block_offset+row, offset, offset<n)
    if tl.sum(((row<n)&(high<=low)).to(tl.int32), 0)>0:
        tl.atomic_max(BAD, 1)


def defer(prediction, target):
    if prediction.cdf.dtype != torch.int16 or target.dtype != torch.int32:
        raise TypeError('Batched coarse descriptor requires int16 CDF and int32 symbols')
    # Own both tensors through launch: graph output and caller labels may be reused.
    # StageExecutor already clones the delivered CDF; target is an indexed tensor.
    return dict(n=len(target), deferred=(prediction.cdf, target))


def finalize(descriptors, bad):
    pending = [d for d in descriptors if 'deferred' in d]
    if not pending:
        return
    total = sum(d['n'] for d in pending)
    blocks = sum((d['n']+31)//32 for d in pending)
    base = torch.empty((total, 2), device=bad.device, dtype=torch.int32)
    lengths = torch.empty(blocks, device=bad.device, dtype=torch.int32)
    offsets = torch.empty_like(lengths)
    metadata = []
    row_at = block_at = 0
    for d in pending:
        q, y = d['deferred']
        n = d['n']
        count = (n+31)//32
        metadata.append([q.data_ptr(), y.data_ptr(), n, row_at, block_at])
        view = base[row_at:row_at+n]
        d.update(base=view, components=view, lengths=lengths[block_at:block_at+count],
                 offsets=offsets[block_at:block_at+count], dynamic=0, k4=0,
                 priors=bad, paid=bad, post=bad)
        row_at += n
        block_at += count
    # One metadata transfer per frame, after all coarse predictions are available.
    meta = torch.tensor(metadata, device=bad.device, dtype=torch.int64)
    _prepare_batch[(len(pending), triton.cdiv(max(d['n'] for d in pending), 128))](
        meta, base, lengths, offsets, bad, 128, num_warps=4)
