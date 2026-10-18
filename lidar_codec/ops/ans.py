"""CUDA bindings for independent 32-lane, bounded-posterior ANS."""

import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import torch

LANES = 32


class EntropyBackend:
    def __init__(self):
        source = Path(__file__).parents[1] / "csrc/block_ans.cu"
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            raise RuntimeError(
                "nvcc is required; set CUDA_HOME and add its bin directory to PATH"
            )
        compiler = subprocess.check_output([nvcc, "--version"], text=True)
        if torch.cuda.get_device_capability() != (8, 9):
            raise RuntimeError(
                "This frozen numeric profile requires compute capability 8.9"
            )
        digest = hashlib.sha256(
            source.read_bytes() + compiler.encode() + b":sm89:lanes32:v1"
        ).hexdigest()
        folder = (
            Path(
                os.environ.get("LIDAR_CODEC_CACHE", Path.home() / ".cache/lidar_codec")
            )
            / digest
        )
        folder.mkdir(parents=True, exist_ok=True)
        binary = folder / "block_ans.so"
        if not binary.exists():
            temporary = folder / f"block_ans.{os.getpid()}.so"
            command = [
                nvcc,
                "-O3",
                "-std=c++17",
                "-arch=sm_89",
                "-DJOINT_LANES=32",
                "--shared",
                "-Xcompiler",
                "-fPIC",
                "--ptxas-options=-v",
                str(source),
                "-o",
                str(temporary),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            (folder / "build.log").write_text(result.stdout + result.stderr)
            result.check_returncode()
            temporary.replace(binary)
            (folder / "build.json").write_text(
                json.dumps(dict(command=command, identity=digest), indent=2)
            )
        self.lib = C.CDLL(str(binary))
        self.lib.joint_decode_cdf16.argtypes = [C.c_void_p, C.c_int] + [C.c_void_p]*3 + [C.c_int] + [C.c_void_p]*3
        self.lib.joint_decode_cdf16.restype = C.c_int
        self.lib.joint_encode_many.argtypes = [C.POINTER(C.c_uint64)] + [C.c_int]*4 + [C.c_void_p]*3 + [C.c_int] + [C.c_void_p]*2
        self.lib.joint_encode_many.restype = C.c_int
        self.lib.joint_decode.argtypes = (
            [C.c_void_p] * 5
            + [C.c_int] * 5
            + [C.c_void_p] * 3
            + [C.c_int]
            + [C.c_void_p] * 4
        )
        self.lib.joint_decode.restype = C.c_int
        self.lib.joint_decode_dense_cdf16.argtypes = self.lib.joint_decode.argtypes
        self.lib.joint_decode_dense_cdf16.restype = C.c_int
        self.lib.joint_prepare.argtypes = (
            [C.c_void_p] * 3 + [C.c_int] * 2 + [C.c_void_p] * 3
        )
        self.lib.joint_encode.argtypes = (
            [C.c_void_p] * 5
            + [C.c_int] * 3
            + [C.c_void_p] * 3
            + [C.c_int]
            + [C.c_void_p] * 2
        )
        self.lib.joint_prepare.restype = self.lib.joint_encode.restype = C.c_int
        self.lib.joint_set_ctas.argtypes = [C.c_int]
        self.lib.joint_set_ctas.restype = None
        self.lib.joint_set_ctas(LANES)

    def decode(self, descriptor, state):
        if descriptor.get("packed_cdf16", False):
            code = self.lib.joint_decode_cdf16(C.c_void_p(descriptor["base"].data_ptr()),
                descriptor["n"], C.c_void_p(state.states.data_ptr()),
                C.c_void_p(state.positions.data_ptr()), C.c_void_p(state.storage.data_ptr()),
                state.capacity, C.c_void_p(descriptor["output"].data_ptr()),
                C.c_void_p(state.error.data_ptr()), C.c_void_p(torch.cuda.current_stream().cuda_stream))
            if code:
                raise RuntimeError(f"ANS coarse decode CUDA error {code}")
            return
        ptr = lambda tensor: C.c_void_p(tensor.data_ptr())
        args = [
            ptr(descriptor[k])
            for k in ("base", "components", "priors", "lengths", "offsets")
        ]
        args += [
            len(descriptor["lengths"]),
            descriptor["dynamic"],
            descriptor["k4"],
            state.active,
            state.carry,
        ]
        args += [
            ptr(state.states),
            ptr(state.positions),
            ptr(state.storage),
            state.capacity,
        ]
        args += [
            ptr(descriptor["output"]),
            ptr(descriptor["posterior"]),
            ptr(state.error),
            C.c_void_p(torch.cuda.current_stream().cuda_stream),
        ]
        decoder = self.lib.joint_decode_dense_cdf16 if descriptor.get("dense_cdf16", False) else self.lib.joint_decode
        code = decoder(*args)
        if code:
            raise RuntimeError(f"ANS decode CUDA error {code}")

    def prepare(self, descriptor, state):
        if not descriptor["k4"]:
            return
        ptr = lambda tensor: C.c_void_p(tensor.data_ptr())
        code = self.lib.joint_prepare(
            *[ptr(descriptor[k]) for k in ("components", "priors", "lengths")],
            len(descriptor["lengths"]),
            state.carry,
            ptr(descriptor["paid"]),
            ptr(descriptor["post"]),
            C.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        if code:
            raise RuntimeError(f"ANS posterior CUDA error {code}")

    def encode(self, descriptor, state):
        ptr = lambda tensor: C.c_void_p(tensor.data_ptr())
        code = self.lib.joint_encode(
            *[
                ptr(descriptor[k])
                for k in ("base", "components", "lengths", "paid", "post")
            ],
            len(descriptor["lengths"]),
            descriptor["k4"],
            state.active,
            ptr(state.states),
            ptr(state.positions),
            ptr(state.storage),
            state.capacity,
            ptr(state.error),
            C.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        if code:
            raise RuntimeError(f"ANS encode CUDA error {code}")

"""One host binding call for a frame's ordered entropy kernels."""

def encode_many(self, descriptors, state, *, prepare=True):
    keys = ('base', 'components', 'lengths', 'paid', 'post', 'priors')
    values = []
    for d in descriptors:
        values.extend(d[key].data_ptr() for key in keys)
        values.extend((len(d['lengths']), d['k4']))
    host = (C.c_uint64 * len(values))(*values)
    code = self.lib.joint_encode_many(host, len(descriptors), int(prepare),
        state.active, state.carry, C.c_void_p(state.states.data_ptr()),
        C.c_void_p(state.positions.data_ptr()), C.c_void_p(state.storage.data_ptr()),
        state.capacity, C.c_void_p(state.error.data_ptr()),
        C.c_void_p(torch.cuda.current_stream().cuda_stream))
    if code:
        raise RuntimeError(f'ANS batch encode CUDA error {code}')


EntropyBackend.encode_many = encode_many
