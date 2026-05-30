#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// CUDA kernel for sigmoid activation
__global__ void sigmoid_kernel(const float* __restrict__ input, 
                               float* __restrict__ output, 
                               int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    for (int64_t i = idx; i < size; i += stride) {
        float x = input[i];
        // Numerically stable sigmoid: 1 / (1 + exp(-x))
        output[i] = 1.0f / (1.0f + expf(-x));
    }
}

// Optimized sigmoid using __fdividef and __expf intrinsics
__global__ void sigmoid_kernel_fast(const float* __restrict__ input, 
                                    float* __restrict__ output, 
                                    int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    for (int64_t i = idx; i < size; i += stride) {
        float x = input[i];
        // Use fast math intrinsics for better performance
        output[i] = __fdividef(1.0f, 1.0f + __expf(-x));
    }
}

torch::Tensor forward(torch::Tensor input) {
    // Ensure input is contiguous and on CUDA
    input = input.contiguous();
    if (!input.is_cuda()) {
        input = input.to(c10::kCUDA);
    }
    
    // Get output tensor (same size as input)
    auto output = torch::empty_like(input);
    
    // Get raw pointers
    const float* input_data = input.data_ptr<float>();
    float* output_data = output.data_ptr<float>();
    
    // Calculate launch parameters
    int64_t size = input.numel();
    const int block_size = 256;  // Optimal for most architectures
    const int grid_size = (size + block_size - 1) / block_size;
    
    // Launch kernel with fast math
    sigmoid_kernel_fast<<<grid_size, block_size>>>(input_data, output_data, size);
    
    // Check for errors
    c10::cuda::CUDAGuard guard(input.device());
    c10::cuda::cudaGetLastError();
    
    return output;
}