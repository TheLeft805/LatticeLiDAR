"""Fixed-capacity geometry graphs, preserving ordered causal slots."""

import torch
from types import SimpleNamespace
from .ops.neighbors import (
    NEAR_OFFSETS,
    FAR_CANDIDATES,
    SUB,
    HISTORY,
    lookup,
    packed_keys,
    history_values,
    _fused_rows,
    _device_offsets,
)
from .geometry import sensor_conditions
from .ops.runtime_fusion import static_outputs, stable_role_rows


class NeighborhoodGraph:
    def __init__(
        self,
        scale,
        capacity,
        device,
        posQ,
        *,
        beam=None,
        history_enabled=True,
        output_workspace=None,
        frame_history=False,
        warm_repeats=2,
        external_pool=None,
        capture_stream=None,
        capture_context=None
    ):
        self.scale, self.capacity = scale, capacity
        self.output_workspace = output_workspace
        self.beam_provider = beam
        self.history_enabled = history_enabled
        self.frame_history = frame_history
        if scale <= 3 and beam is None:
            raise ValueError("Fine fixed geometry requires the original beam provider")
        self.xyz = (torch.zeros((capacity, 3), dtype=torch.int32, device=device) if output_workspace is None else
            output_workspace.view((capacity, 3), torch.int32, device, capacity, ("inputs", "xyz")).zero_())
        self.n = torch.zeros((), dtype=torch.long, device=device)
        self.has_history = torch.zeros((), dtype=torch.bool, device=device)
        self.ref_keys = (torch.full((capacity,), -1, dtype=torch.long, device=device) if output_workspace is None else
            output_workspace.view((capacity,), torch.long, device, capacity, ("inputs", "ref_keys")).fill_(-1))
        self.ref_order = (torch.zeros(capacity, dtype=torch.long, device=device) if output_workspace is None else
            output_workspace.view((capacity,), torch.long, device, capacity, ("inputs", "ref_order")).zero_())
        self.ref_occ = (torch.zeros(capacity, dtype=torch.uint8, device=device) if output_workspace is None else
            output_workspace.view((capacity,), torch.uint8, device, capacity, ("inputs", "ref_occ")).zero_())
        self.conditions = torch.tensor(
            [
                [
                    [[scale, stage, group, expert] for expert in range(6)]
                    for group in range(3)
                ]
                for stage in range(2)
            ],
            device=device,
        )
        self.posQ = posQ
        # Warm the exact synthetic shape, then capture on an auxiliary stream.
        # A shared pool alone is insufficient: cached blocks are stream-local.
        stream = (
            capture_stream
            if capture_stream is not None
            else torch.cuda.Stream(device=device)
        )
        self.capture_stream = stream
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(warm_repeats):
                self.build()
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        self.graph = torch.cuda.CUDAGraph()
        graph_options = {} if external_pool is None else dict(pool=external_pool)
        capture_context = (
            torch.cuda.graph if capture_context is None else capture_context
        )
        with capture_context(self.graph, stream=stream, **graph_options):
            self.output = self.build()
        torch.cuda.synchronize(device)

    def build(self):
        c = self.capacity
        x = self.xyz
        device = x.device
        rows = torch.arange(c, device=device)
        valid = rows < self.n
        anchor = ((x.sum(1) & 1) == 0) & valid
        masks = (
            anchor,
            valid & ~anchor,
            anchor & ((x[:, 0] & 1) == 0),
            anchor & ((x[:, 0] & 1) == 1),
        )
        role_rows, ranks = stable_role_rows(self.output_workspace, c, masks)
        keys, order = torch.sort(torch.where(valid, packed_keys(x), -1))
        invalid = (((x < 0) | (x >= 2**21)).any(1) & valid).any()
        duplicate = ((keys[1:] == keys[:-1]) & (keys[1:] >= 0)).any()
        guards = torch.stack(
            [m.sum() for m in masks] + [invalid.long(), duplicate.long()]
        )
        a, b, a0, a1 = role_rows
        local_a, local_a0 = ranks[0], ranks[2]
        query = x[b]
        nr = self.output_workspace.view((c, 14), torch.long, device, c, ("neighbors", 0)).zero_()
        nm = self.output_workspace.view((c, 14), torch.bool, device, c, ("neighbors", 1)).zero_()
        fr = self.output_workspace.view((c, 8), torch.long, device, c, ("neighbors", 2)).zero_()
        fm = self.output_workspace.view((c, 8), torch.bool, device, c, ("neighbors", 3)).zero_()
        fo = self.output_workspace.view((c, 8, 3), torch.int8, device, c, ("neighbors", 4)).zero_()
        delta = _device_offsets(
            device.index or 0, tuple(NEAR_OFFSETS) + tuple(FAR_CANDIDATES)
        )
        _fused_rows[(c,)](
            query,
            keys,
            local_a[order],
            delta,
            nr,
            nm,
            fr,
            fm,
            fo,
            c,
            query.stride(0),
            query.stride(1),
            c.bit_length(),
            2**21,
            14,
            158,
            8,
            256,
            num_warps=8,
        )
        sub = lookup(x[a1], keys, local_a0[order], SUB)
        if self.history_enabled and self.frame_history:
            history, hm = history_values(
                x, self.ref_keys, self.ref_order, self.ref_occ, self.has_history
            )
        elif self.history_enabled:
            rr, hm = lookup(x, self.ref_keys, self.ref_order, HISTORY, morton=True)
            history = torch.where(
                hm, self.ref_occ[rr], torch.zeros_like(rr, dtype=torch.uint8)
            )
        else:
            # Reset-only protocol: no reference key can match. Preserve exact
            # downstream shapes/arithmetic while eliminating all history lookup.
            hm = torch.zeros((c, len(HISTORY)), dtype=torch.bool, device=device)
            history = torch.zeros((c, len(HISTORY)), dtype=torch.uint8, device=device)
        sensor = sensor_conditions(x, self.scale, self.posQ).float()
        beam = (
            self.beam_provider(x, self.scale, self.posQ)
            if self.scale <= 3
            else torch.zeros((c, 21), device=device)
        )
        paid = torch.full((2, c), -1, dtype=torch.long, device=device)
        static = static_outputs(self.output_workspace, c, (a0, a1, b),
            sensor, history, hm, beam, sub[1], nm, fm, fo, self.has_history, self.scale)
        output = dict(
            rows=rows,
            roles=role_rows,
            guards=guards,
            neighbors=(nr, nm, fr, fm, fo),
            sub=sub,
            history=history,
            history_mask=hm,
            sensor=sensor,
            beam=beam,
            paid=paid,
            static=static,
        )
        return output if self.output_workspace is None else self.output_workspace.store(output, c)

    def replay(self, xyz, level, has_history):
        if not self.history_enabled and (level is not None or has_history):
            raise ValueError("Static absent-history geometry received a reference")
        n = len(xyz)
        self.xyz[:n].copy_(xyz)
        self.xyz[n:].zero_()
        self.n.fill_(n)
        self.has_history.fill_(has_history)
        if self.frame_history and bool(has_history) != (level is not None):
            raise ValueError("Frame history flag and reference disagree")
        if not self.frame_history or has_history:
            self.ref_keys.fill_(-1)
            self.ref_order.zero_()
            self.ref_occ.zero_()
        if level is not None:
            index = level["index"]
            occ = level["occupancy"]
            r = len(occ)
            if r:
                self.ref_keys[-r:].copy_(index.keys)
                self.ref_order[-r:].copy_(index.order)
                self.ref_occ[:r].copy_(occ)
        self.graph.replay()
        out = self.output
        na, nb, n0, n1, invalid, duplicate = out["guards"].tolist()
        if invalid or duplicate:
            raise ValueError("Invalid/duplicate coarse coordinate")
        a, b, a0, a1 = [
            rows[:count] for rows, count in zip(out["roles"], (na, nb, n0, n1))
        ]
        static = [
            {k: v[:count] for k, v in cache.items()}
            for cache, count in zip(out["static"], (n0, n1, nb))
        ]
        return SimpleNamespace(
            all_rows=out["rows"][:n],
            a=a,
            b=b,
            a0=a0,
            a1=a1,
            groups=(a0, a1, b),
            neighbors=tuple(v[:nb] for v in out["neighbors"]) if nb else None,
            sub=tuple(v[:n1] for v in out["sub"]) if n1 else None,
            history=out["history"][:n],
            history_mask=out["history_mask"][:n],
            sensor=out["sensor"][:n],
            beam=out["beam"][:n],
            paid=out["paid"][:, :n],
            static=static,
            conditions=self.conditions,
            posQ=self.posQ,
        )
