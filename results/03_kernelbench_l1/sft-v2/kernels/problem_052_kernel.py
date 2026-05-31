#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void argmin_kernel(const float* input, float* output, int size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int d = dim % dim1;
        output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
    }
}

torch::Tensor argmin_cuda(torch::Tensor x, int dim) {
    auto size = x.numel();
    auto output = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    argmin_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), output.data_ptr<float>(), size, dim);
    return output;
}

torch::Tensor argmin_cpp_source = "torch::Tensor argmin_cuda(torch::Tensor x, int dim);"

torch::Tensor argmin = load_inline(name="argmin", cpp_sources=argmin_cpp_source, cuda_sources=argmin_source, functions=["argmin_cuda"], verbose=True)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(torch::Tensor x):
        return self.argmin.elementwise_add_cuda(x, dim)