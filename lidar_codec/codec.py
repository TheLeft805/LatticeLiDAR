"""Independent-frame geometry codec with explicit shared model and execution profile."""

import hashlib
import json
from pathlib import Path
import struct
import time
from types import SimpleNamespace
import zlib
import numpy as np
import torch
from .model import CompressionModel
from .geometry import Quantization
from .readout import ReadoutExecutor
from .stage import StageExecutor
from .ops.beam import BeamFeatures
from .ops.ans import EntropyBackend
from .ops import cdf as head
from .ops import batched_coarse_entropy
from . import entropy
from .execution import ExecutionPolicy
from .workspace import encoder_capacity, decoder_capacity

PREFIX = 328


class GeometryCodec:
    """Not thread-safe: CUDA graphs share scratch storage within this instance.

    Input is CPU XYZ in meters. Decode returns CPU quantized integer XYZ.
    Model weights, quantization and profile must match at both endpoints.
    A failed frame publishes no points; the next frame starts from fresh state.
    """

    @torch.no_grad()
    def __init__(
        self, checkpoint, *, quantization_mm=16, profile=None, beam_enabled=True
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("The frozen codec requires a CUDA GPU")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        checkpoint = Path(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["schema"] != "lidar-geometry-inference-v1":
            raise ValueError("Unsupported checkpoint schema")
        self.geometry = Quantization(quantization_mm)
        self.model = CompressionModel().cuda().eval().requires_grad_(False)
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.table = SimpleNamespace(
            lookup=saved["prefix_lookup"].cuda(),
            assignments=saved["prefix_assignments"].cuda(),
        )
        self.model.beam = BeamFeatures(
            SimpleNamespace(
                table=saved["beam_table"].cuda(), normal=saved["beam_normal"].cuda()
            )
        )
        self.model.beam.enabled = beam_enabled
        self.model.readout_executor = ReadoutExecutor(self.model.readout)
        self.plan = (
            json.loads(
                (Path(__file__).parents[1] / f"configs/q{quantization_mm}.json").read_text()
            )
            if profile is None
            else profile
        )
        self.stage = StageExecutor(
            self.model,
            self.geometry.posQ,
            self.plan["fixed_stage_capacities"],
            conv_fp16_scales=self.plan.get(
                "fixed_stage_conv_fp16_scales", (1, 2, 3, 4)
            ),
        )
        self.execution = ExecutionPolicy()
        self.lanes = 32
        self.carry = 128
        self.backend = EntropyBackend()
        source = hashlib.sha256()
        root = Path(__file__).parents[1]
        for directory in (root / "lidar_codec", root / "csrc"):
            for path in sorted(directory.rglob("*")):
                if path.suffix in (".py", ".cu", ".cpp", ".cuh", ".h"):
                    source.update(path.relative_to(root).as_posix().encode())
                    source.update(path.read_bytes())
        self.identity = dict(
            schema="lidar-geometry-codec-v1",
            weights_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            source_sha256=source.hexdigest(),
            quantization_mm=quantization_mm,
            beam_enabled=beam_enabled,
            profile=self.plan,
            lanes=self.lanes,
            carry=self.carry,
            posterior_bits=48,
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
        )
        self.prefix = (
            b"LGC1"
            + hashlib.sha256(
                json.dumps(self.identity, sort_keys=True).encode()
            ).digest()
            + bytes(PREFIX - 36)
        )

    def close(self):
        """Release references to captured graphs. Instances must not be reused."""
        self.stage = None
        self.model = None

    def wrap(self, body):
        return self.prefix + struct.pack("<QI", len(body), zlib.crc32(body)) + body

    def unwrap(self, packet):
        if len(packet) < PREFIX + 12 or packet[:PREFIX] != self.prefix:
            raise ValueError("Codec model, numeric profile or quantization mismatch")
        size, crc = struct.unpack_from("<QI", packet, PREFIX)
        body = packet[PREFIX + 12 :]
        if len(body) != size or zlib.crc32(body) != crc:
            raise ValueError("Frame extent or checksum mismatch")
        return body

    def owner(self, symbols, blocks, active, *, capacity=None):
        # A symbol may emit two bytes, plus posterior/prior symbols per block.
        # Deliberately conservative per-lane bound; not a paid fixed-size stream.
        if capacity is None:
            capacity = 2 * (symbols + 2 * blocks) + 16
        return SimpleNamespace(
            carry=self.carry,
            active=int(active),
            capacity=capacity,
            states=torch.full((self.lanes,), 1 << 23, device="cuda", dtype=torch.int32),
            positions=torch.zeros(self.lanes, device="cuda", dtype=torch.int32),
            storage=torch.empty(
                (self.lanes, capacity), device="cuda", dtype=torch.uint8
            ),
            error=torch.zeros((), device="cuda", dtype=torch.int32),
        )

    def pack(self, owner, groups, symbols):
        if self.execution.transfer:
            states, positions, data = self.execution.download(owner)
        else:
            if int(owner.error.item()):
                raise ValueError("Entropy encoder failed")
            states = owner.states.cpu().tolist()
            positions = owner.positions.cpu().tolist()
            extent = max(positions, default=0)
            data = owner.storage[:, :extent].contiguous().cpu().numpy()
        stacks = [
            entropy.LaneState(s, data[k, : positions[k]].tobytes())
            for k, s in enumerate(states)
        ]
        return entropy.pack(
            stacks,
            active=owner.active,
            carry=owner.carry,
            groups=groups,
            symbols=symbols,
        )

    @torch.inference_mode()
    def encode(self, points, trace=False):
        from .geometry import quantize
        from .bitstream import FramePacket, TemporalHeader

        started = time.perf_counter()
        bad = torch.zeros((), device="cuda", dtype=torch.int32)
        xyz = torch.as_tensor(
            quantize(points, self.geometry), device="cuda", dtype=torch.int32
        )
        levels, unique = self.model.backbone.hierarchy(xyz)
        coords = levels[0][0]
        state = None
        base = tuple(tuple(map(int, row)) for row in coords[:, 1:].cpu().tolist())
        descriptors = []
        indices = []
        checks = []
        for depth, (_, occ, child_parent) in enumerate(levels):
            scale = len(levels) - depth
            s0, s1 = occ & 15, occ >> 4
            expert = (
                self.model.backbone.num_experts
                if depth + 2 <= 6
                else self.model.backbone.select_encoder_index(s0, s1)
            )
            indices.append(expert)
            gen = self.stage.predictions(
                coords,
                expert,
                state,
                scale,
                depth,
                scale == 1,
                None,
                posQ=self.geometry.posQ,
                borrow_coarse=self.execution.coarse,
                continuation=(levels[depth+1][0], child_parent) if scale > 1 else (None, None),
            )
            try:
                pred = next(gen)
                while True:
                    y = (s0 if pred.stage == 0 else s1)[pred.rows].contiguous()
                    if scale > 3 and not trace:
                        d = batched_coarse_entropy.defer(pred, y)
                    else:
                        d = head.prepare(
                            pred, coords[pred.rows, 1:] if scale <= 3 else None,
                            scale, self.table, bad, y, trace
                        )
                    descriptors.append(d)
                    if trace:
                        checks.append(
                            dict(
                                labels=y.cpu().numpy(),
                                base=d["trace_base"].cpu().numpy().reshape(len(y), 17),
                                components=(
                                    d["trace_components"].cpu().numpy()
                                    if d["k4"]
                                    else None
                                ),
                                priors=d["priors"].cpu().tolist() if d["k4"] else None,
                            )
                        )
                    try:
                        pred = gen.send(y)
                    except StopIteration as done:
                        coords, state = done.value
                        break
            finally:
                gen.close()
        batched_coarse_entropy.finalize(descriptors, bad)
        if int(bad.item()):
            raise ValueError("Nonfinite CDF")
        symbols = sum(d["n"] for d in descriptors)
        blocks = sum(len(d["lengths"]) for d in descriptors)
        owner = self.owner(symbols, blocks, True, capacity=encoder_capacity([d["n"] for d in descriptors]))
        self.backend.encode_many(descriptors, owner)
        fallback = False
        if int(owner.error.item()):
            owner.active = 0
            owner.error.zero_()
            owner.positions.zero_()
            owner.states.fill_(1 << 23)
            self.backend.encode_many(descriptors, owner, prepare=False)
            fallback = True
        entropy = self.pack(owner, len(descriptors), symbols)
        frame = FramePacket(
            self.geometry.posQ,
            base,
            tuple(indices),
            entropy,
            TemporalHeader(),
        )
        packet = self.wrap(frame.pack())
        elapsed = time.perf_counter() - started
        return packet, dict(
            total_seconds=elapsed,
            bytes=len(packet),
            symbols=symbols,
            fallback=fallback,
            unique_points=unique,
            groups=len(descriptors),
            trace=checks if trace else None,
        )

    @torch.inference_mode()
    def decode(self, packet, trace=False, *, return_device=False):
        from .bitstream import unpack_frame

        started = time.perf_counter()
        frame = unpack_frame(self.unwrap(packet), temporal=True)
        if frame.posQ != self.geometry.posQ:
            raise ValueError("Quantization mismatch")
        if not frame.temporal.reset:
            raise ValueError("Reset-only stream")
        if any(i > self.model.backbone.num_experts for i in frame.indices):
            raise ValueError("Expert outside model")
        parsed = entropy.EntropyFrame(frame.arithmetic, self.carry)
        extent = max(len(s.data) for s in parsed.stacks)
        owner = self.owner(
            parsed.expected[1],
            parsed.expected[1] // 32 + parsed.expected[0],
            parsed.active,
            capacity=decoder_capacity(extent, parsed.expected[1], parsed.expected[0]),
        )
        if self.execution.transfer:
            self.execution.upload(owner, parsed, extent)
        else:
            data = np.zeros((self.lanes, extent), np.uint8)
            for k, s in enumerate(parsed.stacks):
                data[k, : len(s.data)] = np.frombuffer(s.data, np.uint8)
            owner.storage[:, :extent].copy_(torch.from_numpy(data).cuda())
            owner.states.copy_(
                torch.tensor(
                    [s.state for s in parsed.stacks], device="cuda", dtype=torch.int32
                )
            )
            owner.positions.copy_(
                torch.tensor(
                    [len(s.data) for s in parsed.stacks], device="cuda", dtype=torch.int32
                )
            )
        base = torch.tensor(frame.base_coords, device="cuda", dtype=torch.int32)
        coords = torch.cat((torch.zeros_like(base[:, :1]), base), 1)
        state = None
        count = symbols = 0
        checks = []
        for depth, expert in enumerate(frame.indices):
            scale = len(frame.indices) - depth
            gen = self.stage.predictions(
                coords, expert, state, scale, depth, scale == 1, None, posQ=frame.posQ,
                runtime_error=owner.error if self.execution.checks else None,
                borrow_coarse=self.execution.coarse
            )
            try:
                pred = next(gen)
                while True:
                    if scale > 3 and not trace:
                        d = dict(packed_cdf16=True, base=pred.cdf, n=len(pred.rows),
                                 output=torch.zeros(len(pred.rows), device="cuda", dtype=torch.int16))
                    else:
                        d = head.prepare(
                            pred, coords[pred.rows, 1:] if scale <= 3 else None,
                            scale, self.table, owner.error, trace=trace,
                        )
                    self.backend.decode(d, owner)
                    y = d["output"]
                    count += 1
                    symbols += len(y)
                    if trace:
                        previous = torch.cat((torch.zeros_like(y[:1]), y[:-1])).long()
                        previous[torch.arange(len(y), device="cuda") % 32 == 0] = 0
                        base = (
                            d["base"][torch.arange(len(y), device="cuda"), previous, 0]
                            if d["dynamic"]
                            else d["base"]
                        )
                        comp = (
                            d["components"][
                                torch.arange(len(y), device="cuda"), previous
                            ]
                            if d["k4"]
                            else None
                        )
                        if d.get("dense_cdf16", False):
                            base = base.int() & 65535
                            base[..., -1] = 65536
                            if comp is not None:
                                comp = comp.int() & 65535
                                comp[..., -1] = 65536
                        checks.append(
                            dict(
                                labels=y.cpu().numpy(),
                                base=base.cpu().numpy(),
                                components=(
                                    comp.cpu().numpy() if comp is not None else None
                                ),
                                priors=d["priors"].cpu().tolist() if d["k4"] else None,
                            )
                        )
                    del d  # ANS reads precede subsequent allocations on the same stream.
                    try:
                        pred = gen.send(y)
                    except StopIteration as done:
                        if not self.execution.checks and int(owner.error.item()):
                            raise ValueError("Poisoned frame before layer continuation")
                        coords, state = done.value
                        break
            finally:
                gen.close()
        if (count, symbols) != parsed.expected:
            raise ValueError("Declared extent differs")
        if not bool(((owner.states == (1 << 23)) & (owner.positions == 0)).all()):
            raise ValueError("Noncanonical terminal states")
        quantized = coords[:, 1:]
        if not return_device:
            quantized = quantized.cpu().numpy()
        return quantized, dict(
            total_seconds=time.perf_counter() - started,
            groups=count,
            symbols=symbols,
            canonical=True,
            trace=checks if trace else None,
        )
