"""Six sparse experts, causal anchor attention, shared readout and rank-four mixture."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from .layers import (
    OctantEmbedding,
    SparseResidualBlock,
    OccupancyPredictor,
    OccupancyEmbedding,
)
from .ops import morton as mt
from .ops.neighbors import NEAR_OFFSETS


class TokenReadout(nn.Module):
    def __init__(self, query_size, token_size, width):
        super().__init__()
        self.query = nn.Sequential(
            nn.LayerNorm(query_size), nn.Linear(query_size, width)
        )
        self.token = nn.Sequential(
            nn.LayerNorm(token_size), nn.Linear(token_size, width)
        )
        self.mix = nn.Linear(width, width)
        self.output = nn.Linear(width, 16)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, query, tokens, mask, reliability=None, *, tokens_projected=False):
        if self.training or torch.is_grad_enabled():
            raise ValueError(
                f"Anchor attention requires frozen inference: training={self.training}, grad={torch.is_grad_enabled()}"
            )
        if not hasattr(self, "_coarse_half"):
            self._coarse_half = {
                id(t): t.half()
                for layer in (self.query[1], self.mix, self.output)
                for t in (layer.weight, layer.bias)
            }

        def linear(x, layer):
            y = F.linear(
                x.half(),
                self._coarse_half[id(layer.weight)],
                self._coarse_half[id(layer.bias)],
            )
            return y

        q = linear(self.query[0](query), self.query[1])
        projected = tokens if tokens_projected else self.token(tokens)
        t = projected.half()
        # These activation buffers are newly produced and have one consumer.
        # No input token, model tensor, reduction order or GEMM shape is changed.
        v = F.silu(q[:, None, :] + t, inplace=True)
        v = F.silu(linear(v, self.mix), inplace=True)
        weights = mask.float()
        if reliability is not None:
            weights = weights * reliability.float()
        count = weights.sum(1, keepdim=True)
        weighted = v.float()
        weighted.mul_(weights[:, :, None])
        pooled = weighted.sum(1) / count.clamp(min=1)
        return linear(pooled, self.output).float() * (count > 0)


class GeometryExpert(nn.Module):
    def __init__(self, channels=32, kernel_size=3, context_type="none"):
        super().__init__()

        self.local_net = OctantEmbedding(channels)  # Octant positional embedding
        self.feat_resnet0 = SparseResidualBlock(
            channels, kernel_size
        )  # Feature refinement before stage-1 coding
        self.feat_resnet1 = SparseResidualBlock(
            channels, kernel_size
        )  # Feature refinement before stage-1 coding
        self.pred_net0 = OccupancyPredictor(channels)  # Stage-1 occupancy prediction
        self.pred_net1 = OccupancyPredictor(channels)  # Stage-2 occupancy prediction
        self.prior_net0 = OccupancyEmbedding(
            channels
        )  # Stage-1 occupancy context embedding
        self.prior_net1 = OccupancyEmbedding(
            channels
        )  # Stage-2 occupancy context embedding
        self.prior_s0_resnet0 = SparseResidualBlock(
            channels, kernel_size
        )  # Feature refinement after stage-1 coding
        self.prior_s0_resnet1 = SparseResidualBlock(
            channels, kernel_size
        )  # Feature refinement after stage-1 coding
        self.prior_s1_resnet0 = SparseResidualBlock(
            channels, kernel_size
        )  # Feature refinement after stage-2 coding
        self.prior_s1_resnet1 = SparseResidualBlock(
            channels, kernel_size
        )  # Feature refinement after stage-2 coding

        # Channel-wise blending weights between features propagated from level b-1 and current level b
        self.blend_weights = nn.Parameter(
            torch.full((2, channels), 1.0, dtype=torch.float32)
        )  # [2, C]


class EntropyReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_norm = nn.LayerNorm(32)
        self.base = nn.Linear(32 + 16 + 17 + 5, 128)
        # Concatenation gives every ordered slot its own projection columns.
        self.evidence = nn.Linear(30 * 18 + 27 * 9 + 24 + 21 + 4, 128)
        self.conditions = nn.ModuleList([nn.Embedding(n, 128) for n in (22, 2, 3, 6)])
        self.norm = nn.LayerNorm(128)
        self.film = nn.Linear(128, 256)
        self.update = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 128))
        self.output = nn.Linear(128, 16)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        base = torch.cat(
            (
                self.feature_norm(x["features"]),
                F.log_softmax(x["bare"], dim=-1),
                F.one_hot(x["own"].long(), 17).float(),
                x["sensor"],
            ),
            1,
        )
        h = self.base(base)
        token = F.one_hot(x["tokens"].long(), 17).float()
        token = torch.cat((token, x["mask"].float().unsqueeze(-1)), 2).flatten(1)
        bits = (
            (x["history"].long().unsqueeze(-1) >> torch.arange(8, device=h.device)) & 1
        ).float()
        history = torch.cat((bits, x["history_mask"].float().unsqueeze(-1)), 2).flatten(
            1
        )
        e = self.evidence(
            torch.cat(
                (
                    token,
                    history,
                    x["offsets"].float().flatten(1) / 3.0,
                    x["beam"],
                    x["available"].float(),
                ),
                1,
            )
        )
        condition = sum(
            embedding(x["condition"][:, i].long())
            for i, embedding in enumerate(self.conditions)
        )
        gamma, beta = self.film(condition).chunk(2, 1)
        for step in range(2):
            update_input = self.norm(h) * (1 + gamma) + beta + e
            h = h + self.update(update_input)
        return x["bare"] + self.output(self.norm(h))


class BlockMixtureHead(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.project = nn.Linear(128, rank, bias=False)
        self.component = nn.Linear(rank, 64)
        self.prior = nn.Linear(128, 4)

    def forward(self, h, base, mask):
        delta = self.component(self.project(h)).reshape(*h.shape[:2], 4, 16)
        q = F.log_softmax(base[:, :, None, :] + delta, -1)
        pooled = (h * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
        return q, F.log_softmax(self.prior(pooled), -1)


class AnchorAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor_embedding = nn.Embedding(17, 32)
        self.readout = TokenReadout(37, 35, 32)
        self.register_buffer(
            "anchor_offsets14",
            torch.tensor(NEAR_OFFSETS, dtype=torch.int32),
            persistent=False,
        )
        self.sa_token_cache = True
        self.sa_metadata_cache = False
        self.sa_token_table = None

    @torch.no_grad()
    def prepare_sa_token_cache(self):
        if self.sa_token_table is not None:
            return self.sa_token_table
        if self.training:
            raise RuntimeError("Anchor cache requires frozen evaluation mode")
        symbols = torch.arange(17, device=self.anchor_embedding.weight.device)
        values = self.anchor_embedding(symbols)[:, None, :].expand(-1, 14, -1)
        positions = self.anchor_offsets14.to(values.dtype)[None, :, :].expand(
            17, -1, -1
        )
        self.sa_token_table = self.readout.token(torch.cat((values, positions), -1))
        self.sa_token_slot_ids = torch.arange(14, device=values.device)
        torch.cuda.synchronize()


class GeometryBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 5
        self.register_buffer("centers", torch.zeros(5, 32))
        self.experts = nn.ModuleList([GeometryExpert() for _ in range(6)])

    def select_encoder_index(self, x_O_s0, x_O_s1):
        x_H_s0 = torch.bincount(x_O_s0.int().view(-1), minlength=16)
        x_H_s1 = torch.bincount(x_O_s1.int().view(-1), minlength=16)
        x_H = torch.cat([x_H_s0, x_H_s1], dim=0).float()
        x_H = x_H / x_H.sum()
        dists = torch.norm(self.centers - x_H[None, :], dim=1)
        idx = torch.argmin(dists).item()
        return idx

    def compute_bitdepth(self, coords):
        return math.ceil(math.log2(coords[:, 1:].max().item() + 1))

    def hierarchy(self, xyz):
        """Create labels once; no batch mixing or Morton wraparound is supported."""
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
            raise ValueError("Expected nonempty quantized Nx3 integer geometry")
        if xyz.dtype not in (torch.int32, torch.int64) or bool(
            ((xyz < 0) | (xyz >= 2**21)).any()
        ):
            raise ValueError("Geometry must be nonnegative 21-bit integers")
        coords = torch.cat((torch.zeros_like(xyz[:, :1]), xyz), dim=1).int()
        coords, keys = mt.morton3_sort(coords)
        unique_count = len(coords)
        levels = []
        # Integer right shifts reduce the maximum bit depth by one per level.
        # Read it once rather than synchronizing the device at every level.
        depth = max(1, int(coords[:, 1:].max().item()).bit_length() - 2)
        for _ in range(depth):
            coords, keys, occupancy, child_parent = mt.down_once(coords, keys)
            levels.append((coords, occupancy, child_parent))
        return levels[::-1], unique_count


class CompressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = GeometryBackbone()
        self.anchor = AnchorAttention()
        self.readout = EntropyReadout()
        self.mixture = BlockMixtureHead(4)
        self.joint_k4_active = True

    @property
    def device(self):
        return self.backbone.centers.device
