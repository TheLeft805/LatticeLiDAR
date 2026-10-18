"""Experimental decoder reconstruction, with a distinct asset-bound LGC identity.

Retains the existing 328-byte identity envelope and entropy body. The learned
head is pre-shared like the frozen model; its file hash and this implementation
are committed by the identity prefix. LSQ2 framing remains fully chargeable.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


class ReconstructionCodec:
    def __init__(self, checkpoint, head_path, *, kind, quantization_mm, profile):
        from lidar_codec.codec import GeometryCodec, PREFIX
        if kind not in ('separate', 'shared', 'point64', 'gelu64', 'beam61'):
            raise ValueError('Unsupported reconstruction head')
        self.base = GeometryCodec(checkpoint, quantization_mm=quantization_mm, profile=profile)
        self.geometry = self.base.geometry
        self.terminal = None
        self.decoding = False
        self.last_quantized = None
        self.residual = None
        self.w = self.b = None
        self.beam_w = self.beam_b = None
        dependencies = {}
        with np.load(head_path) as heads:
            if kind in ('point64', 'gelu64', 'beam61'):
                from residual_head import ResidualHead
                names = ('affine_w', 'affine_b', 'mean', 'std', 'hidden.weight', 'hidden.bias', 'output.weight', 'output.bias')
                state = {name: torch.tensor(heads[f'{kind}_{quantization_mm}_{name}'], dtype=torch.float32) for name in names}
                if any(not torch.isfinite(value).all() for value in state.values()):
                    raise ValueError('Nonfinite residual reconstruction asset')
                mode = 'gelu64' if kind == 'beam61' else kind
                self.residual = ResidualHead(*(state[name] for name in names[:4]), mode)
                self.residual.load_state_dict(state, strict=True)
                self.residual = self.residual.cuda().eval()
                dependencies['residual_head.py'] = hashlib.sha256(Path(__file__).with_name('residual_head.py').read_bytes()).hexdigest()
                if kind == 'beam61':
                    w, b = heads[f'{kind}_{quantization_mm}_beam_w'], heads[f'{kind}_{quantization_mm}_beam_b']
                    if w.shape != (61,24) or b.shape != (24,) or not np.isfinite(w).all() or not np.isfinite(b).all():
                        raise ValueError('Invalid folded beam residual asset')
                    self.beam_w = torch.tensor(w, device='cuda', dtype=torch.float32)
                    self.beam_b = torch.tensor(b, device='cuda', dtype=torch.float32)
            else:
                w = heads[f'{kind}_{quantization_mm}_w']
                b = heads[f'{kind}_{quantization_mm}_b']
                if w.shape != (8,40,3) or b.shape != (8,3) or not np.isfinite(w).all() or not np.isfinite(b).all():
                    raise ValueError('Invalid folded reconstruction asset')
                self.w = torch.tensor(w.transpose(1,0,2).reshape(40,24),device='cuda',dtype=torch.float32)
                self.b = torch.tensor(b.reshape(24),device='cuda',dtype=torch.float32)
        self.children = torch.arange(8,device='cuda')
        if kind == 'beam61':
            output = 'float32 XYZ meters; frozen GELU64 clipped, then existing-beam linear residual and second half-cell clip; no temporal feedback'
        elif self.residual is not None:
            output = f'float32 XYZ meters; per-child affine plus {kind} residual clipped to half cell; no temporal feedback'
        else:
            output = 'float32 XYZ meters; per-child affine offset clipped to half cell; no temporal feedback'
        self.identity = dict(schema='lidar-reconstruction-codec-v1',base=self.base.identity,
            head_sha256=hashlib.sha256(Path(head_path).read_bytes()).hexdigest(),kind=kind,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            output=output)
        if dependencies:
            self.identity['dependencies'] = dependencies
        self.prefix = b'LGR1'+hashlib.sha256(json.dumps(self.identity,sort_keys=True).encode()).digest()+bytes(PREFIX-36)
        for bucket in self.base.stage.buckets[1]:
            self._observe_terminal(bucket)

    @property
    def last_quantized(self):
        # Optional diagnostic; CPU XYZ is the required reconstruction output.
        if isinstance(self._last_quantized, torch.Tensor):
            self._last_quantized = self._last_quantized.cpu().numpy()
        return self._last_quantized

    @last_quantized.setter
    def last_quantized(self, value):
        self._last_quantized = value

    def _observe_terminal(self, bucket):
        original = bucket.predictions
        def predictions(item):
            result = yield from original(item)
            if self.decoding:
                features, occupancy, _ = bucket.out[6][bucket.capacity]
                n = len(item['coords'])
                # References only: no diagnostic copies or synchronization.
                beam = bucket.geometry.output['beam'][:n] if self.beam_w is not None else None
                self.terminal = (features[:n],occupancy[:n],result[0][:,1:],beam)
            return result
        bucket.predictions = predictions

    def encode(self, raw):
        started = time.perf_counter()
        packet, stats = self.base.encode(raw)
        packet = self.prefix+packet[len(self.prefix):]
        stats = dict(stats, total_seconds=time.perf_counter()-started)
        return packet, stats

    @torch.inference_mode()
    def decode(self, packet):
        started = time.perf_counter()
        if packet[:len(self.prefix)] != self.prefix:
            raise ValueError('Reconstruction model, assets or profile differ')
        self.decoding, self.terminal, self.last_quantized = True, None, None
        try:
            quantized, stats = self.base.decode(self.base.prefix+packet[len(self.prefix):], return_device=self.base.execution.output)
            if self.terminal is None:
                raise RuntimeError('Missing final decoded state')
            f, occ, leaves, beam = self.terminal
            occupied = (occ.long()[:,None] >> self.children) & 1
            features = torch.cat((f,occupied.float()),1)
            parent = torch.repeat_interleave(torch.arange(len(f),device='cuda'),occupied.sum(1),output_size=len(quantized))
            child = (leaves[:,0].long()&1) | ((leaves[:,1].long()&1)<<1) | ((leaves[:,2].long()&1)<<2)
            if self.residual is None:
                offset = (features@self.w+self.b).reshape(-1,8,3).clamp(-.5,.5)[parent,child]
            else:
                offset = self.residual(features,parent,child)
            if self.beam_w is not None:
                correction = (torch.cat((features,beam),1)@self.beam_w+self.beam_b).reshape(-1,8,3)[parent,child]
                offset = (offset+correction).clamp(-.5,.5)
            center = ((leaves.double()*self.geometry.posQ-131072)*.001).float()
            xyz = (center+offset*(self.geometry.posQ*.001)).cpu().numpy()
            self.last_quantized = quantized
            return xyz, dict(stats,total_seconds=time.perf_counter()-started)
        finally:
            self.decoding, self.terminal = False, None

    def close(self):
        self.base.close()
        self.terminal = self.last_quantized = self.w = self.b = self.children = None
        self.residual = None
        self.beam_w = self.beam_b = None
