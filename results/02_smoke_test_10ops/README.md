# 02 — Smoke Test: 10 Elementwise Ops (DPO model, pass@3)

**Model**: `EvanOLeary/laguna-xs2-dense-k8-cuda-dpo`
**Date**: May 2026
**GPU**: A100 80GB
**Script**: `scripts/eval_10ops_isolated.py`

## What this tests
Correctness of the DPO-trained model on 10 unary elementwise operations using subprocess-isolated evaluation. pass@3 with temperature sampling.

## Settings
| Param | Value |
|---|---|
| temperature | 0.6 |
| top_k | 20 |
| max_new_tokens | 1024 |
| pass@k | 3 (best-of-3) |
| evaluation | subprocess-isolated (`scripts/eval_worker.py`) |
| system prompt | CUDA-specific with API hints (scalar_type, AT_DISPATCH, bounds guard) |

## Prompt format
```
System: "You are an expert GPU kernel engineer for PyTorch 2.7 / CUDA 12.8. Write correct CUDA kernels.
- Use `input.scalar_type()` (NOT `input.type()`); dispatch with AT_DISPATCH_FLOATING_TYPES.
- Bounds guard `if (idx < size)`; output = `torch::empty_like(input)`.
- Define `torch::Tensor forward(torch::Tensor input)` and end with a PYBIND11_MODULE binding."

User: "Convert this PyTorch module into an optimized CUDA kernel:
```python
import torch
import torch.nn as nn
class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)
```"
```

## Results (best-of-3)

| Op | Compiled | Correct | Speedup vs eager |
|---|---|---|---|
| **ReLU** | 2/3 | **2/3** | 0.93x |
| **Tanh** | 2/3 | **2/3** | — |
| Sigmoid | 0/3 | 0/3 | — |
| GeLU | 1/3 | 0/3 | — |
| Abs | — | — | — |
| SiLU | — | — | — |
| Softplus | — | — | — |
| ELU | — | — | — |
| LeakyReLU | — | — | — |
| Mish | — | — | — |

## Key findings
1. ReLU and Tanh work at ~70% pass@3 — consistent across runs
2. Sigmoid/GeLU fail due to **float4 vectorization cast bugs** (`float4* v = float4* x;` instead of `reinterpret_cast<float4*>(x)`)
3. No single op beats eager (memory-bandwidth-bound) — speedups require fusion
4. The API-hint system prompt helps (vs no hints: lower compile rate)

## Failure taxonomy
| Category | Ops affected | Fix |
|---|---|---|
| float4 cast bugs | Sigmoid, GeLU | RFT compile reward |
| Deprecated API (`input.type()`) | Various | Prompt hint (already in system prompt) |
| Over-generation (1024 tok exhausted) | Complex ops | raise max_tokens or stop-on-``` |
