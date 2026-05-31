#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel for ReLU activation
__global__ void relu_kernel(const float* __restrict__ input, 
                           float* __restrict__ output, 
                           int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    for (int64_t i = idx; i < num_elements; i += stride) {
        output[i] = fmaxf(input[i], 0.0f);
    }
}

// Optimized CUDA kernel for ReLU gradient (backward pass)
__global__ void relu_backward_kernel(const float* __restrict__ grad_output,
                                    const float* __restrict__ input,
                                    float* __restrict__ grad_input,
                                    int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    for (int64_t i = idx; i < num_elements; i += stride) {
        grad_input[i] = grad_output[i] * (input[i] > 0.0f ? 1.0f : 0.0f);
    }
}

torch::Tensor forward(torch::Tensor input) {
    // Ensure contiguous input
    input = input.contiguous();
    
    // Get output tensor (same size as input)
    torch::Tensor output = torch::empty_like(input);
    
    if (input.is_cuda()) {
        // CUDA implementation
        int64_t num_elements = input.numel();
        
        if (num_elements == 0) return output;
        
        // Optimal block size (256 is typically best for ReLU)
        const int block_size = 256;
        const int grid_size = (num_elements + block_size - 1 + 7) / (block_size * 8); // Limit grid size
        
        dim3 block(block_size);
        dim3 grid(min(grid_size, 65535)); // CUDA limit for grid dimensions
        
        const float* input_ptr = input.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();
        
        relu_kernel<<<grid, block>>>(input_ptr, output_ptr, num_elements);
        
        // Check for CUDA errors
        cudaGetLastError();
    } else {
        // CPU fallback using ATen
        output = torch::relu(input);
    }
    
    return output;
}