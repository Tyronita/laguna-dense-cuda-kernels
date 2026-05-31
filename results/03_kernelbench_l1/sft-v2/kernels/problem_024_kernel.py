import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void logsoftmax_kernel(const float* input, float* output, int batch_size, int dim) {
    extern __shared__ float shared_mem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < batch_size * dim; i += stride) {
        int row = i / dim;
        int col = i % dim;
        float max_val = -FLT_MAX;
        for (int j = 0; j < dim; j++) {
            float val = input[i + j];
            if (val > max_val) {
                max_val = val;
            }
        }
        shared_mem[threadIdx.x] = max_val;
        __syncthreads();
        float exp_val = 0.0f;
        for (int j = 0; j < dim; j++) {
            exp_val += exp_val * (input[i + j] - shared_mem[threadIdx.x]);
        }
        shared_mem[threadIdx.x] = exp_val;
        __syncthreads();
        float sum = 0.0f;
        for (int j = 0; j < dim; j++) {
            sum += shared_mem[threadIdx.x] * (input[i + j] + 1e-5f);
        }
        output[i] = sum / (dim + 1e-5f);
    }
}

torch::Tensor logsoftmax_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.size(1) == x.size(2), "Input must have same dim");

    int batch_size = x.size(0);
    int dim = x.size(1);
    auto output = torch::empty_like(x);

    const int threads = 256;
    const int blocks = (batch_size * dim + threads - 1) / threads;
    size_t shared_mem_size = threads * sizeof(float);

    logsoftmax_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );

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
