import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

torch::Tensor layer_norm_cuda(torch::Tensor x, weight, bias) {
    const float eps = 1e-5;
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = x.data_ptr<float>();

    // Compute mean and variance
    float mean = 0.0f;
    float var = 0.0f;
    for (int i = 0; i < C; i++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float val = input_ptr[i * H * W + h * W + w];
                mean += val;
                var += val * val;
            }
        }
    }

    // Normalize
    float normalized = (x - mean) * (sqrtf(var + eps));

    // Apply weight and bias
    for (int i = 0; i < C; i++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float val = normalized[i * H * W + h * W + w];
                output_ptr[i * H * W + h * W + w] = 
                    weight[i].cast<float>() * val + bias[i].cast<float>();
            }
        }
    }

    return x;
}

torch::Tensor forward(torch::Tensor x, weight, bias) {
    return layer_norm_cuda(x, weight, bias);
}"""

cpp_source = """torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor layer_norm_cuda(torch::Tensor x, weight, bias);\ntorch::Tensor forward(torch::Tensor x, weight, bias);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['elementwise_add_cuda', 'layer_norm_cuda', 'forward'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.elementwise_add_cuda(x)
