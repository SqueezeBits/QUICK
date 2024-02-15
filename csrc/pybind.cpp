#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "gemm_cuda_quick.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_forward_cuda_quick", &gemm_forward_cuda_quick, "QUICK AWQ GEMM kernel.");
}
