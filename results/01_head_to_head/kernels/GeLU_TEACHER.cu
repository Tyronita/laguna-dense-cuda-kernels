#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    }

// Optimized GELU kernel using the approximate formula: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
__global__ void gelu_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Use grid-stride loop for better occupancy with large tensors
    for (int i = idx; i < size; i += stride) {
        float x = input[i];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/π)
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        
        // Fast tanh approximation using __tanhf intrinsic
        float tanh_val = tanhf(inner);
        output[i] = 0.5f * x * (1.0f + tanh_val);
    }
}

// Alternative precise GELU using error function (more accurate but slightly slower)
__global__ void gelu_exact_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        float x = input[i];
        // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        float erf_val = erff(x / sqrtf(2.0f));
        output[i] = 0.5f * x * (1.0f + erf_val);
    }
}

// CUDA kernel launcher
void gelu_cuda_launcher(const float* input, float* output, int size) {
    // Optimal block size for most architectures
    const int block_size = 256;
    const int max_blocks = 65535;
    
    // Calculate grid size with cap to avoid launch failures
    int grid_size = (size + block_size - 1) / block_size;
    grid_size = min(grid_size, max_blocks);
    
    gelu_kernel<<<grid_size, block_size>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
}

// Main forward function
torch::Tensor forward(torch::Tensor input) {
    // Ensure contiguous input
    input = input.contiguous();
    
    // Get tensor dimensions
    auto input_size = input.sizes();
    int64_t num_elements = input.numel();
    
    // Handle empty tensor
    if (num_elements == 0) {
        return input.clone();
    }
    
    // Create output tensor
    torch::Tensor output = torch::empty_like(input);
    
    // Get raw pointers
    float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    // Launch kernel
    gelu_cuda_launcher(input_ptr, output_ptr, num_elements);
    
    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return output;
}

// PyTorch extension definitions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU forward function");
}