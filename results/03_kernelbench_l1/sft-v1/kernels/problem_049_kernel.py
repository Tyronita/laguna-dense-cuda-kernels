#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void max_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int64_t size,
    const int64_t dim) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float max_val = input[idx];
        output[idx] = max_val;
    }
}

torch::Tensor max_reduce_cuda(torch::Tensor input, int64_t dim) {
    auto size = input.numel();
    auto output = torch::empty({1}, input.options());
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    max_reduce_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        dim
    );
    
    return output;
}

torch::Tensor max_reduce_cpp_source = "torch::Tensor max_reduce_cuda(torch::Tensor input, int64_t dim)"

torch::Tensor max_reduce = load_inline(
    name="max_reduce",
    cpp_sources=std::cpp_max_reduce_cpp_source,
    cuda_sources=std::cuda_max_reduce_kernel,
    functions=["max_reduce_cuda",
    verbose=1
)

class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.max_reduce = max_reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_reduce.max_reduce_cuda(x, self.dim)