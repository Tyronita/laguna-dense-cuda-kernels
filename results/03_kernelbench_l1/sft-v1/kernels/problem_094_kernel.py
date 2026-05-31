#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void mean_squared_error_kernel(const float* __restrict__ input, const float* __restrict__ target, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] - target[idx]) * 2.0f;
    }
}

torch::Tensor mean_squared_error_cuda(torch::Tensor input, torch::Tensor target) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    mean_squared_error_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        target.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    return output;
}

torch::Tensor mean_squared_error_cpp_source = "torch::Tensor mean_squared_error_cuda(torch::Tensor input, torch::Tensor target)"

torch::Tensor mean_squared_error = load_inline(name="mean_squared_error", cpp_sources=mean_squared_error_cpp_source, cuda_sources=mean_squared_error_cuda, functions=["mean_squared_error_cuda", verbose=True])

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean_squared_error = mean_squared_error

    def forward(self, input, target):
        return self.mean_squared_error.mean_squared_error_cuda(input, target)