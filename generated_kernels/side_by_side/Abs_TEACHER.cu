#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

torch::Tensor forward(torch::Tensor input) {
    // Ensure input is on CUDA
    if (!input.is_cuda()) {
        AT_ERROR("Input tensor must be on CUDA device");
    }
    
    // Get input dimensions
    auto input_sizes = input.sizes();
    int64_t num_elements = input.numel();
    
    // Create output tensor with same size and type as input
    torch::Tensor output = torch::empty_like(input);
    
    // Get raw pointers
    float* input_data = input.data_ptr<float>();
    float* output_data = output.data_ptr<float>();
    
    // Launch CUDA kernel
    dim3 block(min(256, num_elements));
    dim3 grid((num_elements + block.x - 1) / block.x);
    
    dim3 grid_dim(grid.x, grid.y, grid.z);
    dim3 block_dim(block.x, block.y, block.z);
    
    void* kernel_args[] = {&input_data, &output_data, &num_elements};
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    cudaLaunchKernelExC(
        &abs_kernel,
        &kernel_args,
        sizeof(kernel_args),
        stream,
        0,
        nullptr
    );
    
    return output;
}

// CUDA Kernel
__global__ void abs_kernel(float* input, float* output, int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = fabsf(input[idx]);
    }
}