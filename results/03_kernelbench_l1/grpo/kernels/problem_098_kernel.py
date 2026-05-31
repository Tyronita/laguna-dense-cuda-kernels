import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fusedKernel(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float* __restrict__ output,
    int size,
    float inv_std,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float y = target[idx];
        float log_val = fmaxf(1.0f, x * (x + eps));
        output[idx] = (y - x) * log_val / (x + inv_std);
    }
}

torch::Tensor fused_kl_div_cuda(
    torch::Tensor input,
    torch::Tensor target,
    float inv_std,
    float eps,
    bool reduction
) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fusedKernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        target.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        inv_std,
        eps
    );
    
    if (reduction) {
        output = output.mean();
    }
    return output;
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor target,
    float inv_std,
    float eps,
    bool reduction
) {
    return fused_kl_div_cuda(input, target, inv_std, eps, reduction);
}"""

cpp_source = """torch::Tensor fused_kl_div_cuda(
    torch::Tensor input,
    torch::Tensor target,
    float inv_std,
    float eps,
    bool reduction
);\ntorch::Tensor forward(
    torch::Tensor input,
    torch::Tensor target,
    float inv_std,
    float eps,
    bool reduction
);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_kl_div_cuda', 'forward'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return custom_ops.fused_kl_div_cuda(predictions, targets)
