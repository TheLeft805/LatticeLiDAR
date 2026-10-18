"""FP16 GEMMs with the qualified FP32 packing, normalization and residual order."""

import torch
from torch.nn import functional as F
from .ops.readout import slot_projection, norm_film
from .ops.runtime_fusion import silu_half


class ReadoutExecutor:
    def __init__(self, model):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("Readout executor requires frozen weights")
        self.model = model
        self.half_parameters = {}
        self.capture_hidden = True
        device = model.output.weight.device
        cond = torch.cartesian_prod(
            *(torch.arange(n, device=device) for n in (22, 2, 3, 6))
        )
        embedding = sum(layer(cond[:, i]) for i, layer in enumerate(model.conditions))
        self.film_table = model.film(embedding).contiguous()
        self.gamma = torch.empty(256, device=device)
        self.slot_weight = (
            model.evidence.weight[:, :540]
            .reshape(128, 30, 18)
            .permute(1, 2, 0)
            .contiguous()
        )
        self.rest_weight = model.evidence.weight[:, 540:].contiguous()
        self.bias = []
        for group in range(3):
            active = (
                set(range(22, 30))
                if group == 1
                else set(range(22)) if group == 2 else set()
            )
            folded = model.evidence.bias.clone()
            for slot in range(30):
                if slot not in active:
                    folded = folded + self.slot_weight[slot, 16]
            self.bias.append(folded)

    def _compute(self, buffers, group):
        # Pointwise kernels promote inputs to FP32 before arithmetic.
        def linear(value, weight, bias):
            # First reached during warm-up, before CUDA graph capture. All
            # source tensors remain owned by the frozen executor/model.
            for tensor in (weight, bias):
                if id(tensor) not in self.half_parameters:
                    self.half_parameters[id(tensor)] = tensor.half()
            return F.linear(
                value.half(),
                self.half_parameters[id(weight)],
                self.half_parameters[id(bias)],
            )

        model = self.model
        n = len(buffers["base"])
        h = linear(buffers["base"], model.base.weight, model.base.bias)
        e = (
            buffers["cached_projection"]
            if "cached_projection" in buffers
            else linear(buffers["rest"], self.rest_weight, self.bias[group])
        )
        buffers["static_projection"] = e
        if group:
            projected = torch.empty_like(e, dtype=torch.float32)
            slot_projection[(n,)](
                e,
                buffers["tokens"],
                buffers["mask"],
                self.slot_weight,
                projected,
                n,
                22 if group == 1 else 0,
                8 if group == 1 else 22,
                num_warps=4,
                enable_fp_fusion=False,
            )
            e = projected
        residual = h
        for step in range(2):
            value = torch.empty_like(h, dtype=torch.float16)
            next_h = torch.empty_like(h, dtype=torch.float32)
            norm_film[(n,)](
                h,
                residual,
                e,
                self.gamma,
                model.norm.weight,
                model.norm.bias,
                value,
                next_h,
                n,
                model.norm.eps,
                step > 0,
                True,
                value, False,
                num_warps=4,
                enable_fp_fusion=False,
            )
            h = next_h
            residual = linear(
                silu_half(linear(value, model.update[0].weight, model.update[0].bias)),
                model.update[2].weight,
                model.update[2].bias,
            )
        value, next_h = torch.empty_like(h), torch.empty_like(h)
        half_value = torch.empty_like(h, dtype=torch.float16)
        norm_film[(n,)](
            h,
            residual,
            e,
            self.gamma,
            model.norm.weight,
            model.norm.bias,
            value,
            next_h,
            n,
            model.norm.eps,
            True,
            False,
            half_value, True,
            num_warps=4,
            enable_fp_fusion=False,
        )
        if self.capture_hidden:
            buffers["hidden"] = value
            self.compute_hidden = value
        return torch.add(buffers["bare"], linear(half_value, model.output.weight, model.output.bias), out=buffers.get("output"))
