"""Original TorchSparse FP32 implicit GEMM with current-stream launches."""

from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def load_native():
    import os
    from torch.utils.cpp_extension import load

    source = Path(__file__).parents[1] / "csrc/sparse"
    os.environ.setdefault("MAX_JOBS", "2")
    return load(
        name="lidar_sparse_stream_v1",
        sources=[
            str(source / "binding.cpp"),
            str(source / "convolution_forward_implicit_gemm_cuda.cu"),
        ],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=True,
    )
