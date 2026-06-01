import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hinge_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float product = predictions[idx] * targets[idx];
        out[idx] = fmaxf(1.0f - product, 0.0f);
    }
}

__global__ void sum_kernel(const float* input, float* output, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    float sum = 0.0f;
    while (i < size) {
        sum += input[i];
        i += gridSize;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s >= 1; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto out = torch::zeros({size}, predictions.options());
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    hinge_loss_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), out.data_ptr<float>(), size);
    
    auto sum = torch::tensor(0.0f, predictions.options());
    const int sum_block_size = 256;
    const int sum_num_blocks = (size + sum_block_size - 1) / sum_block_size;
    sum_kernel<<<sum_num_blocks, sum_block_size, sum_block_size * sizeof(float)>>>(out.data_ptr<float>(), sum.data_ptr<float>(), size);
    
    return sum / size;
}
"""
hinge_loss_cpp_source = "torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"
hinge_loss = load_inline(name="hinge_loss", cpp_sources=hinge_loss_cpp_source, cuda_sources=hinge_loss_source, functions=["hinge_loss_cuda"], verbose=True)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hinge_loss = hinge_loss

    def forward(self, predictions, targets):
        return self.hinge_loss.hinge_loss_cuda(predictions, targets)