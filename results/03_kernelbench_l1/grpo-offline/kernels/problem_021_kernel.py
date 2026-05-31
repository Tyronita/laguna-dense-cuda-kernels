#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for elementwise addition with optimized block size
__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// Custom CUDA kernel for sigmoid activation with optimized block size
__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + __fsqrtf(input[idx]));
    }
}

// Custom CUDA function for elementwise addition
torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

// Custom CUDA function for sigmoid activation
torch::Tensor sigmoid_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    sigmoid_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

// C++ interface exposed to PyTorch
torch::Tensor elementwise_add_cpp_source = "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b)"
torch::Tensor sigmoid_cpp_source = "torch::Tensor sigmoid_cuda(torch::Tensor x)"

// Load inline functions
elementwise_add = load_inline(name="elementwise_add", cpp_sources=elementwise_add_cpp_source, cuda_sources=elementwise_add_source, functions=["elementwise_add_cuda", verbose=True])
sigmoid = load_inline(name="sigmoid", cpp_sources=sigmoid_cpp_source, cuda_sources=sigmoid_kernel, functions=["sigmoid_cuda", verbose=True])

// Optimized model with custom CUDA operators
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.elementwise_add = elementwise_add
        self.sigmoid = sigmoid

    def forward(self, x):
        return self.sigmoid(x)