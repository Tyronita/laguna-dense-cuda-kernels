import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for elementwise addition
__global__ void elementwise_add_kernel(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        float* __restrict__ out,
                                        int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// PyTorch wrapper function for elementwise addition
torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel<<<num_blocks, block_size>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    return out;
}

// PyTorch wrapper function for log softmax with elementwise addition
torch::Tensor log_softmax_cuda(torch::Tensor a, float scale) {
    auto size = a.numel();
    auto out = torch::empty_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel<<<num_blocks, block_size>>>(
        a.data_ptr<float>(),
        scale,
        out.data_ptr<float>(),
        size
    );
    return out.logf(out);
}

// PyTorch wrapper function for kl_loss with log softmax and elementwise addition
torch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto input_shape = predictions.size(1);
    auto scale = torch::scalar_tensor(1.0f);
    auto softmax_pred = log_softmax_cuda(predictions, scale);
    auto softmax_target = log_softmax_cuda(targets, scale);
    return torch::mean(softmax_pred * softmax_target, /*dim=*/1);
}

// PyTorch wrapper function for forward
torch::Tensor forward(torch::Tensor a, float scale) {
    return elementwise_add_cuda(a, scale);
}"""

cpp_source = """torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);\ntorch::Tensor log_softmax_cuda(torch::Tensor a, float scale);\ntorch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets);\ntorch::Tensor forward(torch::Tensor a, float scale);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['elementwise_add_cuda', 'log_softmax_cuda', 'kl_div_cuda', 'forward'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return custom_ops.elementwise_add_cuda(predictions, targets)
