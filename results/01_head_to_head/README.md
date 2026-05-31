# 01 — Head-to-Head: Ours (3B Dense) vs Teacher (33B MoE)

**Model**: `laguna-xs2-dense-k8-cuda-sft` (ours) vs `poolside/Laguna-XS.2` (teacher)
**Date**: May 2026
**GPU**: A100 80GB

## What this tests
Speed and correctness comparison between our 3B dense student and the 33B MoE teacher on 6 identical CUDA generation tasks. Same prompt, same harness, sequential load.

## Settings
| Param | Value |
|---|---|
| temperature | 0.6 |
| top_k | 20 |
| max_new_tokens | 1024 |
| enable_thinking | False |
| sampling | do_sample=True (k=1) |
| system prompt | "Expert GPU kernel engineer" (CUDA API hints) |

## Results

| | Ours (3B dense) | Teacher (33B MoE) |
|---|---|---|
| Load time | **3s** | 35s |
| VRAM | **6 GB** | 67 GB |
| Decode speed | **32.1 tok/s** | 25.4 tok/s |
| Correct (k=1) | 0–2/6 (high variance) | 4/6 |
| Speedup vs eager | all <1x | all <1x (best: SiLU 0.87x) |

**Conclusion**: 11x smaller, 12x less VRAM, 26% faster decode. Neither model beats PyTorch eager on memory-bound elementwise ops — speedups require fusion (L2+).

## Files
```
kernels/
├── ReLU_OURS.cu / ReLU_TEACHER.cu
├── Tanh_OURS.cu / Tanh_TEACHER.cu
├── Sigmoid_OURS.cu / Sigmoid_TEACHER.cu
├── GeLU_OURS.cu / GeLU_TEACHER.cu
├── Abs_OURS.cu / Abs_TEACHER.cu
└── SiLU_OURS.cu / SiLU_TEACHER.cu
```
