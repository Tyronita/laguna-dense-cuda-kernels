import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void warpReduceKernel(const float* input, float* output, int size, int dim) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    
    // Each warp processes one element in the reduction dimension
    for (int i = 0; i < dim; i += 32) {
        float sum = 0.0f;
        // Each thread processes multiple elements in the reduction dimension
        for (int offset = lane; offset < dim - i; offset += 32) {
            sum += input[bid * dim + i + offset];
        }
        
        // Warp-level reduction using shuffle operations
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // First thread in warp writes result
        if (lane == 0) {
            shared[warp] = sum;
        }
        __syncthreads();
        
        // Final reduction by first warp
        if (warp == 0) {
            if (tid < blockDim.x / 32) {
                sum = shared[tid];
            }
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (tid == 0) {
                output[bid] = sum;
            }
        }
    }
}

torch::Tensor sum_reduce_cuda(torch::Tensor x, int dim) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.size(2) == dim, "Dimension must match");
    
    int batch_size = x.size(0);
    int dim = x.size(2);
    auto output = torch::zeros({batch_size}, x.options());
    
    const int threads = 256;
    const int warps = threads / 32;
    const int blocks = batch_size;
    
    warpReduceKernel<<<blocks, threads, warps * sizeof(float)>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), 
        batch_size * dim, dim
    );
    
    return output.view({batch_size, 1, -1});
}"""

cpp_source = """torch::Tensor sum_reduce_cuda(torch::Tensor x, int dim);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['sum_reduce_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.sum_reduce_cuda(x)
