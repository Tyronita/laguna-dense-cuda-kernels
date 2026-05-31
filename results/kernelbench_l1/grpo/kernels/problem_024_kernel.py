import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for log softmax with optimized block size
__global__ void logsoftmax_kernel(const float* input, float* output, int batch_size, int dim) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int batch = bid / dim;
    int d = bid % dim;

    float max_val = -FLT_MAX;
    for (int i = tid; i < dim; i += blockDim.x) {
        max_val = fmaxf(max_val, input[batch * dim + i]);
    }
    shared_mem[tid] = max_val;
    __syncthreads();

    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        sum += expf(input[batch * dim + i] - shared_mem[i]);
    }
    shared_mem[tid] = sum;
    __syncthreads();

    float inv_std = rsqrtf(sum + 1e-5f);
    for (int i = tid; i < dim; i += blockDim.x) {
        output[batch * dim + i] = logf(1.0f + expf(input[batch * dim + i] - shared_mem[i] * inv_std));
    }
}

// C++ interface exposed via PyBind11
torch::Tensor logsoftmax_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.size(1) == x.size(2), "Dimension mismatch");

    int batch_size = x.size(0);
    int dim = x.size(1);
    auto output = torch::empty_like(x);

    const int threads = 256;
    const int blocks = batch_size * dim;
    const int shared_mem_size = threads * sizeof(float);

    logsoftmax_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return output;
}"""

cpp_source = """torch::Tensor logsoftmax_cuda(torch::Tensor x);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['logsoftmax_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.logsoftmax_cuda(x)
