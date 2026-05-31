#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { out[idx] = a[idx] + b[idx];
    }
}

__global__ void elementwise_add_relu_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (a[idx] + b[idx]) > 0.0f ? a[idx + b[idx] : 0.0f
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  Circy = (x + b) / (1 + torch.abs(x + b));
    }
}

torch::Tensor elementwise_add_relu_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_relu_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

torch::Tensor elementwise_add_cpp_source = "torch::Tensor elementwise_add_relu_cuda(torch::Tensor a, torch::Tensor b);"

elementwise_add_relu = load_inline(name="elementwise_add_relu", cpp_sources=elementwise_add_cpp_source, cuda_sources=elementwise_add_relu_source, functions=["elementwise_add_relu_cuda"], verbose=True)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add_relu = elementwise_add_relu

    def forward(self, a, b):
        return self.elementwise_add_relu.elementwise_add_relu_cuda(a, b)