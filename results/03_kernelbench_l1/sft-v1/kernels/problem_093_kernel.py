#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void masked_cumsum_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mask,
    float* __restrict__ output,
    const int64_t size,
    const int64_t dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f;
        if (mask[idx]) {
            sum += x[idx] * mask[idx];
        }
        output[idx] = sum;
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int64_t dim) {
    auto size = x.numel();
    auto output = torch::empty({1}, x.options());
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    masked_cumsum_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        mask.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        dim
    );
    
    return output.view({1});
}

torch::Tensor masked_cumsum_cpp_source = "torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int64_t dim);"

torch::Tensor masked_cumsum = load_inline(name="masked_cumsum", cpp_sources=masked_cumsum_cpp_source, cuda_sources=masked_cumsum_kernel, functions=["masked_cumsum_cuda", verbose=True])

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mask_cumsum = masked_cumsum

    def forward(self, x, mask):
        return self.mask_cumsum.maskumsum_cuda(x, mask, self.dim)