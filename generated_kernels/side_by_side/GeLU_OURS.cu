#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel using a grid-stride loop to handle potential workloads
template <typename scalar_t>
__global__ void gelu_kernel_grid_stride(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (; idx < size; idx += stride) {
        scalar_t val = input[idx];
        scalar_t x_cube = val * val;
        scalar_t y = sqrtf(1.0f + tanhf(x_cube));
        output[idx] = 0.5f * val * y;
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_kernel_grid_stride", ([&] {
        gelu_kernel_grid_stride<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU forward (CUDA)");
}