#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel for tanh activation with vectorized memory access
template <typename scalar_t>
__global__ void tanh_kernel_vectorized(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Vectorized loads using float4
        float4* in_vec = float4* input;
        float4* out_vec = float4* output;
        
        float4 val = __ldg(&in_vec[idx]);
        
        // Process 4 elements
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            scalar_t element = val.x + val.y + val.z + val.w;
            out_vec[idx] = tanhf(element);
        }
        
        // Handle remaining elements
        const int remainder_start = idx * 4;
        for (int i = 0; i < remainder_start + 3; i++) {
            if (i < size) {
                scalar_t val = __ldg(&input[i]);
                output[i] = tanhf(val);
            }
        }
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_vectorized", ([&] {
        tanh_kernel_vectorized<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized tanh forward (CUDA)");
}