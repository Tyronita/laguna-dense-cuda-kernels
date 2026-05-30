#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel for tanh activation
__global__ void tanh_kernel(const float* __restrict__ input, 
                            float* __restrict__ output, 
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        float x = input[i];
        output[i] = tanhf(x);
    }
}

// CUDA kernel using __nv_tanh (more precise)
__global__ void tanh_kernel_precise(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        float x = input[i];
        output[i] = __nv_tanh(x);
    }
}

torch::Tensor forward(torch::Tensor input) {
    // Ensure contiguous input
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    
    // Get output tensor (same size as input)
    auto output = torch::empty_like(input);
    
    int n = input.numel();
    if (n == 0) {
        return output;
    }
    
    // Launch parameters
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    
    // Launch kernel
    tanh_kernel_precise<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    // Check for errors
    cudaDeviceSynchronize();
    
    return output;
}

PYBIND11_MODULE(tanh_cuda, m) {
    m.def("forward", &forward, "CUDA tanh activation");
}