#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void maxReductionKernel(const float* input, float* output, int size, int dim) {
    extern __shared__ float sharedMem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    if (idx < size) {
        float max_val = input[idx];
        sharedMem[tid] = max_val;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedMem[tid] += sharedMem[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[bid] = sharedMem[0];
    }
}

torch::Tensor maxReduction_cuda(torch::Tensor x, int dim) {
    auto size = x.numel();
    auto output = torch::zeros_like(x);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    maxReductionKernel<<<blocks, threads, threads * sizeof(float)>>>(x.data_ptr<float>(), output.data_ptr<float>(), size, dim);
    return output;
}

torch::Tensor maxReduction_cpp_source = "torch::Tensor maxReduction_cuda(torch::Tensor x, int dim);"

torch::Tensor maxReduction = load_inline(name="maxReduction", cpp_sources=elementwise_add_cpp_source, cuda_sources=elementwise_add_source, functions=["maxReduction_cuda"], verbose=True)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return maxReduction_cuda(x, self.dim)