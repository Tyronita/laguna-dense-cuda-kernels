import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void optimized_fused_kernel(
    const float* input,
    float* output,
    int size,
    int dim,
    int stride,
    int chunk_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_idx = idx / chunk_size;
    int local_idx = idx % chunk_size;
    
    if (chunk_idx < chunk_size && local_idx < size) {
        // Process elements in chunks
        for (int i = 0; i < chunk_size; i++) {
            int global_idx = chunk_idx * chunk_size + i;
            if (global_idx < size) {
                float val = input[global_idx];
                output[global_idx] = val + 1.0f;
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t dim,
    torch::Tensor out) {
    
    auto size = x.numel();
    const int threads = 256;
    const int chunk_size = 256;
    const int num_blocks = (size + threads - 1) / threads;
    
    optimized_fused_kernel<<<num_blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        size,
        dim,
        0,
        chunk_size
    );
    
    return out;
}"""

cpp_source = """torch::Tensor forward(
    torch::Tensor x,
    int64_t dim,
    torch::Tensor out);"""

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
