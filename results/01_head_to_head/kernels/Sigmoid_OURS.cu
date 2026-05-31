#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for sigmoid activation using grid-stride loop
template <typename scalar_t>
__global__ void sigmoid_forward_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int max_size = size > 0 ? size : 0;
    
    for (; idx < max_size; idx += stride) {
        output[idx] = 1.0f / (1.0f + input[idx]);
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_forward_kernel", ([&] {
        sigmoid_forward_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward (CUDA)");
}