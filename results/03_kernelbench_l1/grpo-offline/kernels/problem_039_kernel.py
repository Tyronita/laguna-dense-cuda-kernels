import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for L2 normalization using warp-level reduction
__global__ void l2_norm_kernel(const float* input, float* output, int size, int dim) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    for (int i = idx; i < size; i += stride) {
        sum += input[i];
    }
    
    // Warp-level reduction using __shfl_down_sync
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread in warp writes to shared memory
    if (tid % warpSize == 0) {
        shared_mem[tid / warpSize] = sum;
    }
    __syncthreads();
    
    // First warp reduces the warp-level sums
    if (tid < (blockDim.x + warpSize - 1) / warpSize) {
        sum = shared_mem[tid];
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (tid == 0) {
            // Compute L2 normalization
            float norm = sum / 2.0f;
            output[bid * dim + tid] = norm;
        }
    }
}

// C++ interface exposed via PyBind11
torch::Tensor l2_norm_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.size(1) == x.size(2), "Dimension mismatch");

    int batch_size = x.size(0);
    int dim = x.size(1);
    int total_elements = batch_size * dim;
    
    auto output = torch::empty_like(x);
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    const int shared_mem_size = (threads + warpSize - 1) / warpSize * sizeof(float);
    
    l2_norm_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        dim
    );
    
    return output;
}"""

cpp_source = """torch::Tensor l2_norm_cuda(torch::Tensor x);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['l2_norm_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.l2_norm_cuda(x)
