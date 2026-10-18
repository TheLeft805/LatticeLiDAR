"""Seven ordered CUDA graph stages with owned outputs and no eager fallback."""

import time
import copy
import torch
import torchsparse.backends
from torchsparse import SparseTensor
from .neighborhood import NeighborhoodGraph as Bucket
from .ops.sparse import load_native
from .ops.readout import pack_inputs
from .prediction import Prediction
from .capture import CaptureSession


class BaseStage:

    def __init__(
        self,
        system,
        posQ,
        scale=10,
        capacity=128,
        expert=5,
        *,
        conv_fp16=False,
        packed_buffers=None,
        state_workspace=None,
        geometry_workspace=None,
        neural_workspace=None,
        geometry_pool=None,
        geometry_stream=None,
        external_pool=None,
        neural_stream=None,
        capture_context=None,
    ):
        setup_start = time.perf_counter()
        self.setup_profile = dict(warm_seconds=0.0, capture_seconds=0.0, graphs=0)
        self.setup_profile["allocated_before_bytes"] = torch.cuda.memory_allocated(
            system.device
        )
        self.setup_profile["reserved_before_bytes"] = torch.cuda.memory_reserved(
            system.device
        )
        if capacity < 128 or capacity % 128:
            raise ValueError("Native sparse kmap requires capacity aligned to 128 rows")
        self.system = system
        self.neural_workspace = neural_workspace
        self.scale = scale
        self.capacity = capacity
        self.expert = expert
        self.core = copy.deepcopy(system.backbone.experts[expert])
        self.native = load_native()
        self.executor = system.readout_executor
        geometry_start = time.perf_counter()
        self.geometry = Bucket(
            scale,
            capacity,
            system.device,
            posQ,
            beam=system.beam if scale <= 3 else None,
            history_enabled=True,
            frame_history=True,
            warm_repeats=2,
            output_workspace=geometry_workspace,
            external_pool=geometry_pool,
            capture_stream=geometry_stream,
            capture_context=capture_context,
        )
        self.setup_profile["geometry_seconds"] = time.perf_counter() - geometry_start
        self.posQ = posQ
        device = system.device
        c = capacity
        self.sa_extra = torch.tensor([scale / 21.0, 0.0], device=device)
        storage = make_stage_state(c, device) if state_workspace is None else state_workspace
        self.state_storage = storage
        for name, (shape, dtype) in stage_state_shapes(c).items():
            count = 1
            for dim in shape:
                count *= dim
            if storage[name].numel() < count or storage[name].dtype != dtype:
                raise ValueError("Stage workspace is too small or has wrong dtype")
            setattr(self, name, storage[name][:count].view(shape))
        self.group_masks = [getattr(self, "group_mask" + str(k)) for k in range(3)]
        self.inverse = [getattr(self, "inverse" + str(k)) for k in range(3)]
        self.graphs = []
        self.out = []
        self.external_fine_cdf = scale <= 3
        self.conv_fp16 = conv_fp16
        self.conv_layers = [m for m in self.core.modules() if hasattr(m, "kernel")]
        self.half_kernels = (
            {id(m): m.kernel.half() for m in self.conv_layers} if conv_fp16 else {}
        )
        self.group_capacities = sorted(
            {max(128, (c // d + 127) // 128 * 128) for d in (8, 4, 2, 1)}
        )
        self.group_calls = {}
        system.anchor.prepare_sa_token_cache()
        if system.anchor.sa_metadata_cache:
            system.anchor.prepare_sa_metadata_cache()
        self.buffers = (
            packed_buffers
            if packed_buffers is not None
            else make_packed_buffers(c, device, 6)
        )
        if len(self.buffers) != 6 or any(
            (len(buf["base"]) < c for buf in self.buffers)
        ):
            raise ValueError("Packed workspace is too small")
        self.gamma = []
        for stage in (0, 1):
            for group in range(3):
                self.gamma.append(
                    self.executor.film_table[
                        ((scale * 2 + stage) * 3 + group) * 6 + expert
                    ].clone()
                )
        self.geometry.n.fill_(1)
        self.geometry.graph.replay()
        capture_context = (
            torch.cuda.graph if capture_context is None else capture_context
        )
        stream = neural_stream if neural_stream is not None else torch.cuda.Stream()
        self.capture_stream = stream
        stream.wait_stream(torch.cuda.current_stream())
        causal_pointers = tuple((t.data_ptr() for t in self.group_masks + self.inverse))
        layer_pool = (
            external_pool
            if external_pool is not None
            else torch.cuda.graph_pool_handle()
        )
        for k in range(7):
            graphs = {}
            outputs = {}
            pool = layer_pool
            for rows_capacity in reversed(self.group_capacities) if k < 6 else [c]:
                warm_start = time.perf_counter()
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self.compute(k, rows_capacity)
                stream.synchronize()
                g = torch.cuda.CUDAGraph()
                self.setup_profile["warm_seconds"] += time.perf_counter() - warm_start
                capture_start = time.perf_counter()
                with capture_context(g, stream=stream, pool=pool):
                    out = self.compute(k, rows_capacity)
                if (
                    tuple((t.data_ptr() for t in self.group_masks + self.inverse))
                    != causal_pointers
                ):
                    raise RuntimeError("Causal graph buffers changed address")
                stream.synchronize()
                graphs[rows_capacity] = g
                outputs[rows_capacity] = out
                self.setup_profile["capture_seconds"] += (
                    time.perf_counter() - capture_start
                )
                self.setup_profile["graphs"] += 1
            self.graphs.append(graphs)
            self.out.append(outputs)
        self.calls = 0
        self.setup_profile["total_seconds"] = time.perf_counter() - setup_start
        self.setup_profile["warm_repeats"] = 2
        self.setup_profile["allocated_after_bytes"] = torch.cuda.memory_allocated(
            device
        )
        self.setup_profile["reserved_after_bytes"] = torch.cuda.memory_reserved(device)

    def conv(self, f, layer):
        if self.conv_fp16:
            out = self.native.conv(
                f.half().contiguous(),
                self.half_kernels[id(layer)],
                self.kmap,
                self.capacity,
                32,
                False,
                True,
            ).float()
        else:
            out = self.native.conv(
                f.contiguous(),
                layer.kernel.contiguous(),
                self.kmap,
                self.capacity,
                32,
                torchsparse.backends.allow_tf32,
                torchsparse.backends.allow_fp16,
            )
        if layer.bias is not None:
            out = out + layer.bias
        return out

    def residual(self, f, block):
        return torch.relu(
            self.conv(torch.relu(self.conv(f, block.conv0)), block.conv1) + f
        )

    def stack(self, f, a, b):
        return self.residual(self.residual(f, a), b)

    def commit(self, k):
        stage, group = divmod(k, 3)
        mask = self.group_masks[group]
        rank = self.inverse[group]
        self.paid[stage].copy_(
            torch.where(mask, self.symbols[k, rank], self.paid[stage])
        )

    def compute(self, k, rows_capacity=None):
        c = self.capacity
        geo = self.geometry.output
        rows = geo["rows"]
        valid = rows < self.geometry.n
        rows_capacity = c if rows_capacity is None else rows_capacity
        if k == 0:
            self.paid.fill_(-1)
            xyz = self.coords[:, 1:]
            anchor = xyz.sum(1) & 1 == 0
            masks = (
                valid & anchor & (xyz[:, 0] & 1 == 0),
                valid & anchor & (xyz[:, 0] & 1 == 1),
                valid & ~anchor,
            )
            for target, value in zip(self.group_masks, masks):
                target.copy_(value)
            for target, mask in zip(self.inverse, self.group_masks):
                target.copy_(mask.long().cumsum(0).sub(1).clamp(min=0))
            f = self.core.local_net(
                SparseTensor(coords=self.coords, feats=self.inherited)
            )
            weights = torch.softmax(self.core.blend_weights, 0)
            f = torch.where(
                self.has_parent, weights[0] * f + weights[1] * self.inherited, f
            )
            f = self.stack(f, self.core.feat_resnet0, self.core.feat_resnet1)
            self.features.copy_(f)
            self.bare.copy_(self.core.pred_net0(f))
        else:
            self.commit(k - 1)
        if k == 3:
            f = self.features + self.core.prior_net0(self.paid[0].clamp(0, 15))
            f = self.stack(f, self.core.prior_s0_resnet0, self.core.prior_s0_resnet1)
            self.features.copy_(f)
            self.bare.copy_(self.core.pred_net1(f))
        if k == 6:
            f = self.features
            if self.scale != 1:
                f = f + self.core.prior_net1(self.paid[1].clamp(0, 15))
                f = self.stack(
                    f, self.core.prior_s1_resnet0, self.core.prior_s1_resnet1
                )
            occ = (self.paid[0] | self.paid[1] << 4).int()
            good = ((self.paid >= 0) & (self.paid <= 15) | ~valid[None]).all() & (
                (occ > 0) | ~valid
            ).all()
            return (f, occ, good)
        stage, group = divmod(k, 3)
        a, b, a0, a1 = geo["roles"]
        r = (a0, a1, b)[group][:rows_capacity]
        x = geo["static"][group]
        tokens = torch.full((rows_capacity, 30), 16, dtype=torch.uint8, device=r.device)
        logits = self.bare[r]
        if group == 1:
            sr, sm = [v[:rows_capacity] for v in geo["sub"]]
            values = self.paid[stage, a0].clamp(0, 15)
            tokens[:, 22:] = torch.where(sm, values[sr], 16).to(torch.uint8)
        if group == 2:
            nr, nm, fr, fm, _ = [v[:rows_capacity] for v in geo["neighbors"]]
            values = self.paid[stage, a].clamp(0, 15)
            tokens[:, :14] = torch.where(nm, values[nr], 16).to(torch.uint8)
            tokens[:, 14:22] = torch.where(fm, values[fr], 16).to(torch.uint8)
            if stage == 0 and "normal" != "skip":
                sa = self.system.anchor
                symbols = self.paid[0, a].clamp(0, 15)[nr]
                if sa.sa_token_cache:
                    sa_tokens = sa.sa_token_table[symbols, sa.sa_token_slot_ids]
                else:
                    source = sa.anchor_embedding(symbols)
                    sa_tokens = torch.cat(
                        (
                            source,
                            sa.anchor_offsets14.to(source.dtype).expand(
                                rows_capacity, -1, -1
                            ),
                        ),
                        2,
                    )
                metadata = torch.cat(
                    (
                        (self.coords[r, 1:] & 1).float(),
                        self.sa_extra.expand(rows_capacity, -1),
                    ),
                    1,
                )
                query = torch.cat((self.features[r], metadata), 1)
                correction = sa.readout(
                    query, sa_tokens, nm, tokens_projected=sa.sa_token_cache
                )
                logits = logits + torch.where(
                    geo["guards"][0] > 0, correction, torch.zeros_like(correction)
                )
                del sa_tokens, metadata, query, symbols, correction
                if not sa.sa_token_cache:
                    del source
        own = x["own0"][:rows_capacity] if stage == 0 else self.paid[0, r].clamp(0, 15)
        buf = {name: value[:rows_capacity] for name, value in self.buffers[k].items()}
        model = self.system.readout
        pack_inputs[rows_capacity,](
            self.features[r],
            logits,
            own,
            x["sensor"],
            tokens,
            x["mask"],
            x["history"],
            x["history_mask"],
            x["offsets"],
            x["beam"],
            x["available"],
            model.feature_norm.weight,
            model.feature_norm.bias,
            buf["base"],
            buf["rest"],
            buf["tokens"],
            buf["mask"],
            buf["bare"],
            x["available"].stride(0),
            model.feature_norm.eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
        self.executor.gamma.copy_(self.gamma[k])
        if self.neural_workspace is not None:
            buf["output"] = self.neural_workspace.work.view(
                (rows_capacity, 16), torch.float32, r.device, rows_capacity, ("prediction", 0))
        out = self.executor._compute(buf, group)
        from .ops import cdf as op

        cdf = None
        if not self.external_fine_cdf:
            probabilities = torch.softmax(out, -1)
            cdf = op._convert_cdf16_scalar(
                torch.cat(
                    (torch.zeros_like(probabilities[:, :1]), probabilities.cumsum(-1)),
                    1,
                )
            )
        required = torch.zeros_like(self.paid, dtype=torch.bool)
        if stage:
            required[0] = valid
        for g in range(group):
            required[stage] |= self.group_masks[g]
        good = (
            ~required | (self.paid >= 0) & (self.paid <= 15)
        ).all() & torch.isfinite(out).all()
        return (out, cdf, good)

    def prepare(self, item):
        coords = item["coords"]
        n = len(coords)
        if n > self.capacity:
            raise ValueError("Bucket overflow")
        if item["expert"] != self.expert:
            self.expert = item["expert"]
            source = self.system.backbone.experts[self.expert]
            torch._foreach_copy_(
                list(self.core.parameters()), list(source.parameters())
            )
            if self.conv_fp16:
                torch._foreach_copy_(
                    [self.half_kernels[id(m)] for m in self.conv_layers],
                    [m.kernel for m in self.conv_layers],
                )
            for k in range(6):
                self.gamma[k].copy_(
                    self.executor.film_table[(self.scale * 6 + k) * 6 + self.expert]
                )
        ref = item["reference"]
        level = ref.level(self.scale) if ref is not None else None
        if level is not None and len(level["occupancy"]) > self.capacity:
            raise ValueError("Reference capacity exceeded")
        layer = self.core.feat_resnet0.conv0
        from torchsparse.nn import functional as sparse_F

        config = layer._config
        if config is None:
            config = sparse_F.conv_config.get_global_conv_config()
        if config is None:
            config = sparse_F.conv_config.get_default_conv_config(
                conv_mode=sparse_F.get_conv_mode(), training=layer.training
            )
        if (
            layer.training
            or layer.transposed
            or layer.generative
            or (layer.kernel_size != (3, 3, 3))
            or (layer.stride != (1, 1, 1))
            or (layer.dilation != 1)
        ):
            raise ValueError(
                "Direct map requires the fixed stride-one inference convolution"
            )
        if config.dataflow != sparse_F.Dataflow.ImplicitGEMM or config.ifsort:
            raise ValueError("Direct map configuration changed")
        km = sparse_F.build_kernel_map(
            coords,
            n,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            None,
            None,
            None,
            config.kmap_mode,
            config.dataflow,
            downsample_mode=config.downsample_mode,
            training=False,
            ifsort=config.ifsort,
            split_mask_num=config.split_mask_num,
            split_mask_num_bwd=config.split_mask_num_bwd,
        )
        if km["reorder_out_in_map"] is not None:
            raise ValueError("Expected original unsorted kmap")
        self.kmap.fill_(-1)
        self.kmap[:n].copy_(km["out_in_map"][:n])
        self.coords[:n].copy_(coords)
        self.coords[n:].zero_()
        self.inherited.zero_()
        self.has_parent.fill_(item["state"] is not None)
        if item["state"] is not None:
            self.inherited[:n].copy_(item["state"]["features"])
        plan = self.geometry.replay(coords[:, 1:], level, ref is not None)
        self.symbols.zero_()
        self.calls += 1
        return plan

    def predictions(self, item):
        if item["last"] != (self.scale == 1):
            raise ValueError("Fixed-stage terminal scale differs")
        plan = self.prepare(item)
        n = len(item["coords"])
        checks = []
        for k in range(6):
            stage, group = divmod(k, 3)
            rows = plan.groups[group]
            capacity = next((c for c in self.group_capacities if c >= len(rows)))
            self.graphs[k][capacity].replay()
            self.group_calls[k, capacity] = self.group_calls.get((k, capacity), 0) + 1
            out, cdf, good = self.out[k][capacity]
            checks.append(good.clone())
            if len(rows):
                logits = out[: len(rows)]
                if not (item.get("borrow_coarse", False) and self.scale > 3):
                    logits = logits.clone()
                delivered_cdf = cdf[: len(rows)] if cdf is not None else None
                if delivered_cdf is not None:
                    delivered_cdf = delivered_cdf.clone()
                symbols = yield Prediction(
                    stage,
                    ("A0", "A1", "B")[group],
                    rows,
                    logits,
                    delivered_cdf,
                    "cdf16-scalar-v1" if cdf is not None else None,
                    checks[-1],
                )
                if len(symbols) != len(rows):
                    raise ValueError("Symbol count differs")
                self.symbols[k, : len(rows)].copy_(symbols)
        self.graphs[6][self.capacity].replay()
        f, occ, good = self.out[6][self.capacity]
        if item.get("runtime_error") is not None:
            checks.append(item["runtime_error"] == 0)
        if not bool(torch.stack(checks + [good]).all()):
            raise ValueError(
                "Invalid occupied parent/commit or causal/finite predicate"
            )
        if item.get("continuation") is not None:
            # Exact Morton child ordering from hierarchy construction. No new
            # conditioning information reaches the predictor; this only copies
            # the already-computed parent features to their known children.
            nc, parent_rows = item["continuation"]
            if item["last"]:
                return None, None
            return nc, dict(features=f[parent_rows], parent=None, context=None)
        from .ops import morton

        if item["last"]:
            return (morton.upscale_coordinate(item["coords"], occ[:n]), None)
        nc, nf = morton.upscale_coordinate_feature(item["coords"], occ[:n], f[:n])
        return (nc, dict(features=nf, parent=None, context=None))


def make_packed_buffers(capacity, device, count):
    """Scratch only: every packed element is written before its stage consumes it."""
    values = [
        dict(
            base=torch.zeros((capacity, 70), device=device, dtype=torch.float16),
            rest=torch.zeros((capacity, 292), device=device, dtype=torch.float16),
            tokens=torch.full((capacity, 30), 16, dtype=torch.uint8, device=device),
            mask=torch.zeros((capacity, 30), dtype=torch.bool, device=device),
            bare=torch.zeros((capacity, 16), device=device),
        )
        for _ in range(count)
    ]
    return values * 6 if count == 1 else values


def stage_state_shapes(c):
    return dict(
        coords=((c, 4), torch.int32),
        inherited=((c, 32), torch.float32),
        has_parent=((), torch.bool),
        kmap=((c, 27), torch.int32),
        symbols=((6, c), torch.long),
        paid=((2, c), torch.long),
        features=((c, 32), torch.float32),
        bare=((c, 16), torch.float32),
        **{"group_mask" + str(k): ((c,), torch.bool) for k in range(3)},
        **{"inverse" + str(k): ((c,), torch.long) for k in range(3)},
    )


def make_stage_state(c, device):
    result = {}
    for name, (shape, dtype) in stage_state_shapes(c).items():
        count = 1
        for dim in shape:
            count *= dim
        result[name] = torch.full(
            (count,), -1 if name in ("kmap", "paid") else 0, dtype=dtype, device=device
        )
    return result


class MixtureStage(BaseStage):

    def __init__(self, *args, **kwargs):
        self.latents = {}
        super().__init__(*args, **kwargs)

    def compute(self, k, rows_capacity=None):
        result = super().compute(k, rows_capacity)
        if self.system.joint_k4_active and self.scale <= 3 and (k < 3):
            h = self.executor.compute_hidden
            n = self.group_masks[k].sum()
            valid = torch.arange(len(h), device=h.device) < n
            blocks = h.reshape(-1, 32, 128)
            mask = valid.reshape(-1, 32)
            head = self.system.mixture
            delta = head.component(head.project(h)).reshape(-1, 4, 16)
            pooled = (blocks * mask[:, :, None]).sum(1) / mask.sum(1).clamp_min(1)[
                :, None
            ]
            probability = head.prior(pooled).softmax(-1)
            probability = probability / probability.sum(-1, keepdim=True)
            prior = torch.floor(124 * probability).to(torch.int32) + 1
            prior.scatter_add_(
                -1,
                prior.argmax(-1, keepdim=True),
                128 - prior.sum(-1, keepdim=True).to(torch.int32),
            )
            if self.neural_workspace is not None:
                delta, prior = self.neural_workspace.mixture(delta, prior, len(h))
            self.latents[k, len(h)] = (delta, prior)
            out, cdf, good = result
            good = (
                good & torch.isfinite(delta).all() & torch.isfinite(probability).all()
            )
            result = (out, cdf, good)
        if self.neural_workspace is not None:
            result = self.neural_workspace.result(k, self.capacity if rows_capacity is None else rows_capacity, result)
        return result

    def predictions(self, item):
        generator = super().predictions(item)
        try:
            pred = next(generator)
            while True:
                delta = prior = None
                if (
                    self.system.joint_k4_active
                    and self.scale <= 3
                    and (pred.stage == 0)
                ):
                    group = ("A0", "A1", "B").index(pred.group)
                    n = len(pred.rows)
                    cap = min((c for c in self.group_capacities if c >= n))
                    delta, prior = self.latents[group, cap]
                    delta = delta[:n]
                    prior = prior[: (n + 31) // 32].clone()
                symbols = yield pred._replace(delta=delta, prior=prior)
                pred = generator.send(symbols)
        except StopIteration as done:
            return done.value
        finally:
            generator.close()


class StageExecutor:
    """Capture once; choose a bounded bucket, commit six groups, then propagate.

    Pools capture larger shapes first; runtime bucket order remains ascending. A caller must
    finish a frame before beginning another one. Output tensors that survive a
    replay are cloned in BaseStage and MixtureStage.
    """

    def __init__(self, model, posQ, capacities, conv_fp16_scales=(1, 2, 3, 4)):
        started = time.perf_counter()
        capacities = {int(s): sorted(set(values)) for s, values in capacities.items()}
        if any(
            not 1 <= s <= 21
            or not values
            or any(type(c) is not int or c < 128 or c % 128 for c in values)
            for s, values in capacities.items()
        ):
            raise ValueError("Invalid aligned capacity profile")
        self.system = model
        self.buckets = {}
        maximum = max(max(v) for v in capacities.values())
        # Sequential replay only; propagation owns its output before reuse.
        from .geometry_workspace import GeometryWorkspace, NeuralWorkspace
        neural_workspace = NeuralWorkspace(maximum, model.device)
        geometry_workspace = GeometryWorkspace(maximum)
        geometry_pool = torch.cuda.graph_pool_handle()
        geometry_stream = torch.cuda.Stream(device=model.device)
        state_workspace = make_stage_state(maximum, model.device)
        buffers = make_packed_buffers(maximum, model.device, 1)
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream(device=model.device)
        self.capture_session = CaptureSession("once")
        with self.capture_session:
            for scale, values in sorted(capacities.items(), key=lambda item: max(item[1]), reverse=True):
                self.buckets[scale] = [
                    MixtureStage(
                        model,
                        posQ,
                        scale,
                        capacity,
                        model.backbone.num_experts,
                        conv_fp16=scale in conv_fp16_scales,
                        packed_buffers=buffers,
                        state_workspace=state_workspace,
                        geometry_workspace=geometry_workspace,
                        neural_workspace=neural_workspace,
                        geometry_pool=geometry_pool,
                        geometry_stream=geometry_stream,
                        external_pool=pool,
                        neural_stream=stream,
                        capture_context=self.capture_session.graph,
                    )
                    for capacity in reversed(values)
                ]
                self.buckets[scale].sort(key=lambda bucket: bucket.capacity)
        self.setup_seconds = time.perf_counter() - started
        self.max_rows = {}
        self.records = []
        self.frame_index = -1

    def predictions(
        self, coords, expert, state, scale, depth, last, reference=None, *, posQ=None, continuation=None, runtime_error=None, borrow_coarse=False
    ):
        if reference is not None:
            raise ValueError("This model codes independent frames")
        if depth == 0:
            self.frame_index += 1
            # Keep only the current frame; previously returned stats stay intact.
            self.records = []
        self.max_rows[scale] = max(self.max_rows.get(scale, 0), len(coords))
        bucket = next(
            (
                b
                for b in self.buckets.get(scale, ())
                if len(coords) <= b.capacity and posQ == b.posQ
            ),
            None,
        )
        if bucket is None:
            raise ValueError(
                f"Frame exceeds frozen execution profile: scale={scale}, rows={len(coords)}, quantization={posQ}"
            )
        self.records.append(
            dict(
                frame_index=self.frame_index,
                scale=scale,
                current_rows=len(coords),
                bucket=bucket.capacity,
                expert=expert,
            )
        )
        return (
            yield from bucket.predictions(
                dict(
                    coords=coords,
                    expert=expert,
                    state=state,
                    scale=scale,
                    depth=depth,
                    last=last,
                    reference=None,
                    posQ=posQ,
                    continuation=continuation,
                    runtime_error=runtime_error, borrow_coarse=borrow_coarse,
                )
            )
        )

    def stats(self):
        return dict(
            setup_seconds=self.setup_seconds,
            capacities={s: [b.capacity for b in bs] for s, bs in self.buckets.items()},
            max_rows=self.max_rows,
            records=self.records,
            capture_session=self.capture_session.stats,
            group_capacity_calls=[
                dict(scale=s, parent_capacity=b.capacity, group=k, capacity=c, calls=n)
                for s, bs in self.buckets.items()
                for b in bs
                for (k, c), n in b.group_calls.items()
            ],
        )
