import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool3d_kernel(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    int kD, int kH, int kW,
    int stride, int padding) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * (D - 1) * (H - 1) * (W - 1);
    if (idx < total) {
        int c = idx / (D * H * W);
        int rem = idx % (D * H * W);
        int d = rem / (H * W);
        int h = rem % (H * W);
        int w = rem / W;
        
        int outD = (D - 1) * stride - 1;
        int outH = (H - 1) * stride - 1;
        int outW = (W - 1) * stride - 1;
        
        float sum = 0.0f;
        for (int kd = 0; kd < kD; kd++) {
            int inD = d + padding - kd;
            if (inD >= 0 && inD < D) {
                for (int kh = 0; kh < kH; kh++) {
                    int inH = h + padding - kh;
                    if (inH >= 0 && inH < H) {
                        for (int kw = 0; kw < kW; kw++) {
                            int inW = w + padding - kw;
                            if (inW >= 0 && inW < W) {
                                int index = c * (D * H * W) + 
                                         inD * (H * W) + 
                                       inH * W + 
                                     inW;
                                sum += input[index];
                            }
                        }
                    }
                }
            }
        }
        output[idx] = sum / (kD * kH * kW);
    }
}

torch::Tensor avg_pool3d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(kernel_size % 3 == 0, "Kernel size must be 3");
    TORCH_CHECK(stride % 3 == 0, "Stride must be 3");
    TORCH_CHECK(padding % 3 == 0, "Padding must be 3");

    int N = x.size(0), C = x.size(1), D = x.size(2), H = x.size(3), W = x.size(4);
    int kD = kernel_size / 2, kH = kernel_size / 3, kW = kernel_size % 3;
    
    int outD = (D - 1) * stride - 1;
    int outH = (H - 1) * stride - 1;
    int outW = (W - 1) * stride - 1;
    
    auto output = torch::empty({N, C, outD, outH, outW}, x.options());
    
    int total = N * C * (D - 1) * (H - 1) * (W - 1);
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    avg_pool3d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        kD, kH, kW,
        stride, padding
    );
    
    return output;
}"""

cpp_source = """torch::Tensor avg_pool3d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding);"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['avg_pool3d_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops.avg_pool3d_cuda(x)
