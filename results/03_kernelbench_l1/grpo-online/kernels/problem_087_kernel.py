import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int height,
    int width,
    int out_channels) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * height * width;
    if (idx < total) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / (width * height)) % out_channels;
        int b = idx / (width * height * out_channels);
        
        float sum = bias[c];
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int h_in = 0; h_in < height; h_in++) {
                for (int w_in = 0; w_in < width; w_in++) {
                    int input_idx = b * (in_channels * height * width) + 
                                   c_in * (height * width) + 
                                   h_in * width + w_in;
                    int weight_idx = c * (in_channels * height * width) + 
                                   c_in * (height * width) + 
                                   h_in * width + w_in;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor conv2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    auto out_channels = weight.size(0);
    
    auto output = torch::empty({batch_size, out_channels, height, width}, x.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * height * width + block_size - 1) / block_size;
    
    conv2d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        height,
        width,
        out_channels
    );
    
    return output;
}

torch::Tensor forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    return conv2d_cuda(x, weight, bias);
}"""

cpp_source = """torch::Tensor conv2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);\ntorch::Tensor forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['conv2d_cuda', 'forward'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.conv2d_cuda(x)
