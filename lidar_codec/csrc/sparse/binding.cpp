#include "convolution_forward_implicit_gemm_cuda.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("conv", &conv_forward_implicit_gemm_cuda); }
