import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void warpReduceKernel(const float* input, float* output, int size, int dim) {
    extern __shared__ float sharedMem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    
    // Each warp processes one element in the reduction dimension
    for (int i = 0; i < dim; i += 32) {
        float sum = 0.0f;
        // Each thread processes multiple elements in the reduction dimension
        for (int offset = i + lane; offset < dim; offset += 32) {
            sum += input[bid * dim + offset];
        }
        
        // Warp-level reduction using shuffle operations
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // Write the reduced sum to shared memory
        if (lane == 0) {
            sharedMem[warp] = sum;
        }
        __syncthreads();
        
        // First warp reduces the warp results
        if (warp == 0) {
            if (lane < 32) {
                sum = sharedMem[lane];
            }
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (lane == 0) {
                output[bid] = sum;
            }
        }
    }
}

torch::Tensor sum_reduce_cuda(torch::Tensor x, int dim) {
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.size(1) == dim, "Dimension must match");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int batch_size = x.size(0);
    int dim = x.size(1);
    auto output = torch::zeros({batch_size}, x.options());

    const int threads = 32;  // 32 threads per block
    const int blocks = batch_size;
    const int shared_mem_size = 32 * sizeof(float);

    warpReduceKernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size * dim,
        dim
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
