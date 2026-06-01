import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose1d_kernel(const float* input, const float* weight, const float* bias, float* output,
                                          int batch_size, int in_channels, int out_channels, int input_length, int output_length,
                                          int kernel_size, int stride, int padding, int dilation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_length;
    if (idx >= total_elements) return;

    int n = idx / (out_channels * output_length);
    int c = (idx / output_length) % out_channels;
    int l = idx % output_length;

    float sum = bias != nullptr ? bias[c] : 0.0f;

    for (int k = 0; k < kernel_size; k++) {
        int input_l = (l + padding) - k * dilation;
        if (input_l >= 0 && input_l < input_length) {
            sum += input[n * in_channels * input_length + c * input_length + input_l] * weight[c * in_channels * kernel_size + k];
        }
    }

    output[idx] = sum;
}

torch::Tensor conv_transpose1d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                     int stride, int padding, int dilation) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int output_length = (input_length - 1) * stride + kernel_size;

    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * output_length + block_size - 1) / block_size;

    conv_transpose1d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, input_length, output_length,
        kernel_size, stride, padding, dilation
    );

    return output;
}
"""

conv_transpose1d_cpp_source = "torch::Tensor conv_transpose1d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int dilation);"

conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size) * 0.01)
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_buffer('bias_param', None)
        
        self.conv_transpose1d = conv_transpose1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias_tensor = self.bias_param if self.bias_param is not None else torch.tensor([])
        return self.conv_transpose1d.conv_transpose1d_cuda(x, self.weight, bias_tensor, self.stride, self.padding, self.dilation)