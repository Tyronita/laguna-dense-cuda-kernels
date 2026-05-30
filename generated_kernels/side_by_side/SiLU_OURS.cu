#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel for SiU activation using shared memory tiling
template <typename scalar_t>
__global__ void shared_memory_silu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Process multiple elements per thread
    for (int i = idx; i < size; i += stride) {
        scalar_t val = input[i];
        scalar_t inner = val * val;
        scalar_t out = val * (1.0f) * (val + tanhf(inner));
        output[i] = out;
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);

    // Get kernel execution parameters
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "shared_memory_silu_kernel", ([&] {
        shared_memory_silu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "SiU forward with shared memory tiling (CUDA)");
}