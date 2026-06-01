import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mse_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mse_kernel(const float* predictions, const float* targets, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        out[idx] = diff * diff;
    }
}

__global__ void reduce_sum_kernel(const float* input, float* output, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    sdata[tid] = 0;
    while (i < size) {
        sdata[tid] += input[i];
        i += gridSize;
    }
    __syncthreads();
    for (int s=blockDim.x/2; s>warpSize/2; s>>=1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

torch::Tensor mse_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto squared_diff = torch::zeros(size, predictions.options());
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    mse_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), squared_diff.data_ptr<float>(), size);
    
    const int sum_block_size = 256;
    const int sum_num_blocks = (size + sum_block_size * 256 - 1) / (sum_block_size * 256);
    auto sum_output = torch::zeros(sum_num_blocks, predictions.options());
    reduce_sum_kernel<<<sum_num_blocks, sum_block_size, sum_block_size * sizeof(float)>>>(squared_diff.data_ptr<float>(), sum_output.data_ptr<float>(), size);
    
    float total_sum = sum_output.sum().item<float>();
    return torch::tensor(total_sum / size, predictions.options());
}
"""

mse_cpp_source = "torch::Tensor mse_cuda(torch::Tensor predictions, torch::Tensor targets);"
mse = load_inline(name="mse", cpp_sources=mse_cpp_source, cuda_sources=mse_source, functions=["mse_cuda"], verbose=True)

class ModelNew(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()
        self.mse = mse

    def forward(self, predictions, targets):
        return self.mse.mse_cuda(predictions, targets)