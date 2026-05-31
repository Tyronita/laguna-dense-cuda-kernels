import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for Tanh activation using warp-level reduction
__global__ void tanh_kernel(const float* input, float* output, int size) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // Compute tanh using fast math
        output[idx] = tanhf(val);
    }
}

// The forward function exposed via PyBind11
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    int size = x.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Launch kernel with shared memory for potential reduction
    tanh_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(),
        x.data_ptr<float>(),
        size
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return x;
}"""

cpp_source = """torch::Tensor forward(torch::Tensor x);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['forward'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.forward(x)
