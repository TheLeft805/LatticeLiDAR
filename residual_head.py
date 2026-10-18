"""Small reconstruction residuals on frozen decoded features; never entropy context.

Point64 and joint32 have identical parameters and nonlinear channels. Only the
second 32-channel branch's spatial reduction changes. The point32 screen comes
first; these modules do not choose a head or launch any experiment.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def prefix_partition(leaves, q_mm, morton_code):
    """Complete leaf partition: fixed 0.512 m cells, at most 32 points per block."""
    if q_mm not in (8, 16, 32, 64) or leaves.ndim != 2 or leaves.shape[1] != 3 or not len(leaves):
        raise ValueError('Expected nonempty decoded leaves at a supported quantization')
    codes = morton_code(leaves)
    order = torch.argsort(codes)
    prefix = codes[order] >> (3 * int(math.log2(512 // q_mm)))
    boundary = torch.cat((torch.ones(1, device=leaves.device, dtype=torch.bool), prefix[1:] != prefix[:-1]))
    starts = torch.nonzero(boundary).flatten()
    sizes = torch.diff(torch.cat((starts, starts.new_tensor([len(leaves)]))))
    prefix_row = boundary.cumsum(0) - 1
    local = torch.arange(len(leaves), device=leaves.device) - starts[prefix_row]
    block_count = (sizes + 31) // 32
    block = (block_count.cumsum(0) - block_count)[prefix_row] + local // 32
    block_starts = torch.nonzero(local % 32 == 0).flatten()
    block_sizes = torch.diff(torch.cat((block_starts, starts.new_tensor([len(leaves)]))))
    return order, block, block_sizes


class ResidualHead(nn.Module):
    """Frozen affine plus zero-start linear/MLP residual, output in voxel units."""
    def __init__(self, affine_w, affine_b, mean, std, mode):
        super().__init__()
        if mode not in ('linear', 'factor32', 'point32', 'point64', 'joint32', 'gelu64'):
            raise ValueError('Unsupported residual head')
        if affine_w.shape != (40, 24) or affine_b.shape != (24,) or mean.shape != (40,) or std.shape != (40,):
            raise ValueError('Invalid frozen affine/normalization shape')
        if not all(torch.isfinite(value).all() for value in (affine_w, affine_b, mean, std)) or not (std > 0).all():
            raise ValueError('Invalid frozen affine/normalization values')
        self.mode = mode
        for name, value in (('affine_w', affine_w), ('affine_b', affine_b), ('mean', mean), ('std', std)):
            self.register_buffer(name, value.detach().clone())
        width = 40 if mode == 'linear' else (32 if mode in ('factor32', 'point32') else 64)
        self.hidden = nn.Identity() if mode == 'linear' else nn.Linear(40, width)
        self.output = nn.Linear(width, 24)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, parent_rows, child, partition=None):
        base = features @ self.affine_w + self.affine_b
        inputs = (features - self.mean) / self.std
        if self.mode == 'linear':
            offsets = base + self.output(inputs)
        else:
            hidden = self.hidden(inputs)
            if self.mode == 'gelu64':
                hidden = F.gelu(hidden, approximate='none')
            elif self.mode != 'factor32':
                hidden = F.relu(hidden)
            if self.mode in ('point64', 'joint32', 'gelu64'):
                own = base + F.linear(hidden[:, :32], self.output.weight[:, :32], self.output.bias)
                context = F.linear(hidden[:, 32:], self.output.weight[:, 32:])
                if self.mode == 'joint32':
                    if partition is None:
                        raise ValueError('Joint residual requires a decoded-geometry partition')
                    order, block, sizes = partition
                    pooled = torch.segment_reduce(context[parent_rows[order]], reduce='mean', lengths=sizes)
                    leaf_block = torch.empty_like(block).index_copy(0, order, block)
                    selected_context = pooled.reshape(-1, 8, 3)[leaf_block, child]
                else:
                    selected_context = context.reshape(-1, 8, 3)[parent_rows, child]
                return (own.reshape(-1, 8, 3)[parent_rows, child] + selected_context).clamp(-.5, .5)
            else:
                offsets = base + self.output(hidden)
        return offsets.reshape(-1, 8, 3)[parent_rows, child].clamp(-.5, .5)
