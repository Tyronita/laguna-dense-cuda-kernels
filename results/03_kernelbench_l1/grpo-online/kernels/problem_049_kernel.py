import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void maxReductionKernel(const float* input, float* output, int batch_size, int dim1, int dim2) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int batch = bid / dim1;
    int dim_idx = bid % dim1;
    
    float max_val = -FLT_MAX;
    
    // Each thread processes multiple elements in the dimension
    for (int i = tid; i < dim2; i += blockDim.x) {
        int idx = batch * dim1 * dim2 + dim_idx * dim2 + i;
        max_val = max(max_val, input[idx]);
    }
    
    // Reduce within the dimension using shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_mem[tid] = max(max_val, max_val + (tid + stride));
        }
        __syncthreads();
        
        if (tid == 0) {
            max_val = shared_mem[0];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[batch * dim1 + dim_idx] = max_val;
    }
}

torch::Tensor maxReduction_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int batch_size = x.size(0);
    int dim1 = x.size(1);
    int dim2 = x.size(2);
    
    auto output = torch::zeros({batch_size, dim1}, x.options());
    
    const int threads = 256;
    const int blocks = batch_size * dim1;
    const int shared_mem_size = threads * sizeof(float);
    
    maxReductionKernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim1,
        dim2
    );
    
    return output;
}"""

cpp_source = """torch::Tensor maxReduction_cuda(torch::Tensor x);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['maxReduction_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.maxReduction_cuda(x)
