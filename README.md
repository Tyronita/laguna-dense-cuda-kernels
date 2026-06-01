# Laguna-Dense — CUDA Kernel Generation

A **~3.0 B fully-dense** model that generates **CUDA / Triton GPU kernels** from PyTorch modules,
**densified from the 33 B [poolside/Laguna-XS.2](https://huggingface.co/poolside/Laguna-XS.2) MoE**.

> Part of the **[laguna-xs2-expert-coactivation-scheduling](https://github.com/cm2435/laguna-xs2-expert-coactivation-scheduling)**
> project (MoE→dense densification). This repo collects **only the CUDA-kernel** work:
> motivation → densification → kernel-mixture pretraining → CUDA SFT → eval harness → results.

📊 Full graph index: **[docs/GRAPHS.md](docs/GRAPHS.md)** · 📓 Ablation log: **[docs/ABLATIONS.md](docs/ABLATIONS.md)**

---

## TL;DR
| | |
|---|---|
| **Teacher** | `poolside/Laguna-XS.2` — 33 B total / 3 B active MoE (256 experts top-8 + shared) |
| **Student** | **~3.0 B dense** (1 SwiGLU FFN per layer, K=8 width) — 5.99 GB bf16 |
| **Method** | DO-ACP warm-start → reconstruction (kernel mix) → CUDA SFT → GRPO/DPO |
| **vs teacher** | **11× fewer params, 12× less VRAM, +26 % faster decode** (32.1 vs 25.4 tok/s) |
| **Status** | SFT + GRPO + DPO done; **10% correct on L1** (3B student) vs **24% teacher**; densification gap identified |

## Pipeline / lineage
```
poolside/Laguna-XS.2 (33B/3B-active MoE, 256 experts)
   │ densify: routed experts → 1 dense SwiGLU (K=8) per layer
   │ Stage 0  DO-ACP warm-start (Gram log-det select 8 experts → concat)
   │ Stage 1  teacher-forced reconstruction on a KERNEL mixture  → "V2" checkpoint
   │ Stage 2  SFT on SakanaAI CUDA                                → cuda-sft / cuda-sft-v2
   │ Stage 3a GRPO-offline (Dr.GRPO on Sakana dataset rewards)    → cuda-grpo
   │ Stage 3b GRPO-online  (Dr.GRPO with live compilation)       → cuda-rft   ← BEST
   ▼ Stage 3c DPO (preference pairs from Sakana traces)           → cuda-dpo
```

## Models (Hugging Face)
| Model | Stage | Repo |
|---|---|---|
| Dense reconstruction (kernel mix) | pretrain (V2) | `EvanOLeary/laguna-xs2-dense-k8-kernelmix` |
| **CUDA-SFT** | SFT | `EvanOLeary/laguna-xs2-dense-k8-cuda-sft` |
| (sibling) Dense reconstruction (Python) | pretrain (V1) | `EvanOLeary/laguna-xs2-dense-k8-recon` |

---

## 1 · Motivation — MoE expert activation (why densify)
Before collapsing the MoE we measured how many of Laguna's **256 routed experts** actually fire on
**C4** (161,932 tokens, all 39 sparse layers):

| Metric | Value |
|---|---|
| Experts ever used | **256 / 256** (100 %) |
| **Effective experts / layer** | **~158** of 256 |
| Mean per-layer coverage | 99.7 % |
| Load Gini (concentration) | **0.53** (peaks mid-stack) |

The routed FFN behaves far **denser** than its 256-way capacity → a dense surrogate is viable, and
**K must exceed top-8**. This motivated **K=8 + DO-ACP warm-start**.

![expert activation](docs/figures/expert_activation.png)

Full analysis: [gist](https://gist.github.com/Tyronita/fb28e9c31c2b66cccb70fbd939bd1c43) · `docs/reports/expert-activation-c4.md`.

---

## 2 · Architecture (output model)
**2,996,678,656 params (~3.0 B), 5.99 GB bf16.** Each sparse layer's 256-expert MoE → **one dense
SwiGLU FFN** (width K8×512 = 4096) + the kept shared expert. Attention/embeddings/norms copied from teacher.

| Component | Params | Trained? |
|---|---|---|
| `routed_dense` × 39 | **0.98 B** | ✅ (reconstruction + SFT) |
| attention × 40 (48/8 GQA, 30 SWA + 10 global) | 1.43 B | ❄️ frozen |
| embed + lm_head | 0.41 B | ✅ SFT only (lm_head) |
| shared experts × 39 | 0.12 B | ❄️ frozen |
| **Total** | **3.00 B** | |
Hidden 2048 · 40 layers · 262 k ctx · 100 352 vocab · SiLU/SwiGLU.

---

## 3 · Training data
### Reconstruction (kernel mixture, "V2")
| Dataset | Weight | Language | Role |
|---|---|---|---|
| `GPUMODE/KernelBook` | 40 % | Python→Triton | kernel |
| `nvidia/OpenCodeInstruct` | 30 % | Python | general code |
| `SakanaAI/AI-CUDA-Engineer-Archive` | 20 % | PyTorch→CUDA-C++ | kernel |
| Triton multiturn traces | 10 % | Triton reasoning | kernel |

≈ **50 % kernel / 30 % Python / 20 % CUDA-C++**.

### SFT (CUDA)
| | |
|---|---|
| Dataset | `SakanaAI/AI-CUDA-Engineer-Archive` (~30,615 rows, CC-BY-4.0) |
| Fields | `PyTorch_Code_Module` (prompt) → `CUDA_Code` (target), filtered `Correct==True` |
| Format | chat: `system + user(PyTorch ```python```) → assistant(```cpp CUDA```)`, prompt masked |
| Not used | `CUDA_Speedup_Native`, `NCU_Profile`, `Clang_Tidy` → reserved for the RFT reward |

---

## 4 · Pretraining — reconstruction (kernel mixture, V2)
Teacher-forced, all-39-layer-parallel reconstruction of each MoE block's output:
`loss = mean_ℓ( MSE/mean(yℓ²) + 0.05·(1−cos) )`, Adafactor 2e-4, only `routed_dense` trained.

| | Value |
|---|---|
| Steps / tokens | **2000 / ~8.2 M** |
| Loss | 0.67 → **0.16** (normalized) |
| Deep-layer MSE (L28-39) | 0.20 → **0.018** (tighter than the Python flavour's 0.022) |
| Hardware | 1× H100, ~35 min, ~77 GB |

![v2 reconstruction](docs/figures/v2_training.png)
![v2 per-layer heatmap](docs/figures/v2_layer_heatmap.png)

(Smoke test — 8 layers: loss 0.049→0.033, cosine 0.95→0.58 — `docs/figures/kernelmix_smoke_curves.png`.)

---

## 5 · SFT — CUDA kernel generation
| | Value |
|---|---|
| Base | V2 kernel-mixture checkpoint |
| Steps / tokens | **400 / ~3.5 M** |
| Trainable | `routed_dense` + `lm_head` + norms (**1.19 B**) |
| Optimizer | AdamW 1e-5, grad-clip 1.0, grad-accum 8, seq 2048 |
| **Result** | **CE 0.675 → 0.21**; emits working CUDA + restores chat format |

![sft curve](docs/figures/sft_curve.png)

**Sample (ReLU, chat prompt):** the model returns a correct CUDA kernel —
```cpp
__global__ void relu_kernel(const float* __restrict__ in, float* __restrict__ out, int64_t n){
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i<n) out[i] = in[i]>0 ? in[i] : 0; }
torch::Tensor forward(torch::Tensor x){ auto o=torch::empty_like(x); int t=256,b=(x.numel()+t-1)/t;
  relu_kernel<<<b,t>>>(x.data_ptr<float>(), o.data_ptr<float>(), x.numel()); return o; }
```

### Training overview (data + steps)
![overview](docs/figures/training_overview.png)

| Stage | Steps | Tokens | Data | Trainable |
|---|---|---|---|---|
| Warm-start (DO-ACP) | — | — | calibration | — |
| Reconstruction (V2) | 2000 | ~8.2 M | kernel mixture | routed_dense |
| SFT (CUDA) | 400 | ~3.5 M | SakanaAI CUDA | routed_dense + lm_head + norms |

---

## 6 · Inference settings (reproducible)
| Knob | Value |
|---|---|
| temperature / top_k | 0.6 / 20 |
| max_new_tokens | 1024 (don't under-cap — truncates complex kernels) |
| do_sample | True (→ use **pass@k**; same prompt gives a different kernel each sample) |
| enable_thinking | False |
| system prompt | must match target DSL (CUDA-only for CUDA, Triton-only for Triton) |

---

## 7 · Results

### 7a · KernelBench Level 1 — all model variants (pass@1, greedy, A100 80GB)

Full evaluation on [KernelBench](https://github.com/ScalingIntelligence/KernelBench) Level 1
(100 single-operator problems: matmul, conv, activations, norms, pooling, reductions, losses).
Greedy decoding (temperature=0), single attempt per problem, subprocess-isolated evaluation.

| Model | Params | Method | Compile | Correct (fast_0) | Faster (fast_1) | Avg Speedup |
|---|---|---|---|---|---|---|
| **Teacher** | **33B MoE** | (baseline) | 57% | **24%** | **4%** | 3.1x |
| **GRPO-online** | 3B dense | Dr.GRPO + live compilation | 23% | 10% | 1% | 14.6x* |
| **GRPO-offline** | 3B dense | Dr.GRPO on Sakana rewards | 19% | 9% | 1% | 8.5x* |
| **DPO** | 3B dense | DPO on Sakana preferences | 27% | 2% | 0% | 0.55x |
| **SFT-v2** | 3B dense | SFT (level 1+2+3, +500 steps) | 21% | 0% | 0% | — |
| **SFT-v1** | 3B dense | SFT (level 1+2, 400 steps) | 27% | 0% | 0% | — |

*Avg speedup inflated by P12 (diagonal matmul, 72-143x algorithmic optimization — legitimate).

**Key findings:**
1. **The 33B MoE teacher scores 24% correct** — significantly better than all 3B dense students.
   Densification lost kernel-writing capability that GRPO only partially recovered on the narrow
   set of ops it was trained on. The teacher uniquely solves 17 ops the best student cannot
   (Sigmoid, Softmax, Swish 1.58x, Softsign 2.24x, ELU, HardSigmoid, L1Norm, etc).
2. **RL is essential for the dense student.** SFT models compile (21-27%) but achieve 0% correctness.
   GRPO reward (compile + correct + speedup) is what teaches the 3B model to produce correct kernels.
3. **The student wins on matmul variants** (P1, P10, P11) — the ops GRPO was trained on —
   but fails on everything else the teacher can do. More diverse SFT/GRPO data is the clear next step.

### 7b · Comparison to frontier models (from [KernelBench paper](https://arxiv.org/abs/2502.10517))

| Model | Size | fast_1 (L1) | fast_1 (L2) | fast_1 (L3) |
|---|---|---|---|---|
| DeepSeek R1 | 671B | 12% | 36% | 2% |
| OpenAI o1 | ~200B | 10% | 24% | 12% |
| Claude 3.5 Sonnet | ~175B | 10% | 7% | 8% |
| DeepSeek V3 | 671B | 6% | 4% | 0% |
| GPT-4o | ~200B | 4% | 5% | 0% |
| Llama 3.1-405B | 405B | 3% | 0% | 0% |
| Llama 3.1-70B | 70B | 3% | 0% | 0% |
| **Ours (GRPO-online, 3B)** | **3B** | **1%** | — | — |

Our 3B model achieves 1% fast_1, comparable to Llama-3.1-405B (3%) despite being **135x smaller**.
The KernelBench L1 ceiling scales weakly with model size — the bottleneck is training data coverage
(which ops the model has seen CUDA for), not raw model capacity.

### 7c · Correct kernels (detail)

| Problem | Op | Speedup | Notes |
|---|---|---|---|
| P1 | Square matmul | 0.26x | Shared-memory tiling, correct but slower than cuBLAS |
| P6 | Large-K matmul | 0.12x | Same tiling, slower on large K dimension |
| P8 | Irregular matmul | 0.24x | Handles M/K/N properly |
| **P12** | **Diagonal matmul** | **72-143x** | **Algorithmic win: exploits diag structure** |
| P13 | Symmetric matmul | 0.26x | Standard tiling (doesn't exploit symmetry) |
| P15 | Lower-triangular matmul | 0.27x | Standard tiling |
| P19 | ReLU | 0.92x | Near PyTorch parity (memory-bound) |
| P22 | Tanh | 0.88x | Uses tanhf(), correct |
| P26 | GELU | 0.85x | Approximation formula, correct |

**P12 is the standout** — the model recognized `diag(A) * B` is elementwise row-scaling, not full
matmul. 72-143x speedup vs PyTorch's broadcast multiply. Passes all 5 correctness trials.

### 7d · Error analysis (GRPO-online, best model)

| Error | Count | Root cause |
|---|---|---|
| Correct | 10 | Working CUDA kernel |
| Compiled but incorrect | 13 | Wrong math (index bugs, wrong formula) |
| `__init__` missing args | 43 | Conv/norm/pool need constructor weights — model only knows elementwise |
| CUDA build error | 15 | Invalid C++ (but compiles simpler float* code) |
| CUDA illegal memory | 5 | Out-of-bounds (flat 1D index into multi-dim tensor) |
| No code extracted | 4 | Model output think-loop or truncated |
| Other | 10 | Eval crash, syntax error |

**The #1 blocker is coverage** (43/100): the model was trained on elementwise ops only (SakanaAI data)
and has no knowledge of conv/pooling/norm CUDA kernels. More diverse training data would directly
address this gap.

### 7e · Speed & size vs teacher (head-to-head, same 6 CUDA questions)
| | OURS (3.0 B dense) | TEACHER Laguna-XS.2 |
|---|---|---|
| Params | **3.0 B** | 33.4 B |
| VRAM / load | **6 GB / 3 s** | 67 GB / 35 s |
| **Decode speed** | **32.1 tok/s** | 25.4 tok/s |
→ **11x smaller, 12x less VRAM, +26% faster.** Neither model beats PyTorch eager on single
elementwise ops (memory-bandwidth-bound — speedups need fusion / KernelBench L2).

### 7f · Earlier smoke test (DPO model, 10 ops, pass@3) — for context
| Op | pass@3 correct | speedup vs eager |
|---|---|---|
| **ReLU** | 2/3 (67%) | 0.93x |
| **Tanh** | 2/3 (67%) | — |
| Sigmoid | 0/3 | fails |

Different setup: system prompt with API hints, temperature=0.6, best-of-3. Consistent with the
KernelBench L1 results (ReLU/Tanh work, harder ops don't).

### What's in KernelBench (the benchmark)
| Level | # | Contents |
|---|---|---|
| L1 | 100 | single ops — mostly **matmul/conv** + activations/norms/reductions/losses |
| L2 | 100 | **fusion** chains (where >1x speedup is winnable) |
| L3 | 50 | full nets (ResNet/VGG/DenseNet) |
| L4 | 20 | HF-model-level |

---

## 8 · ⚠️ Reproducibility finding — isolate kernel evaluation
Running generated CUDA **in the model's process is INVALID**: a buggy kernel (out-of-bounds write)
corrupts the CUDA context and makes **every later eval fail**, regardless of the model →
order-dependent, contaminated results. **Compile + run each kernel in its own subprocess**
(`scripts/eval_worker.py`). Verified: a crashing kernel segfaults only the worker; the driver survives.
(KernelBench / robust-kbench do the same.)

## 9 · Failure taxonomy (from generated CUDA)
| Category | Example | Fix |
|---|---|---|
| Wrong math/formula | GeLU / Sigmoid / Softmax | RFT correctness reward |
| Deprecated API | `input.type()` vs `.scalar_type()` | prompt hint / RFT |
| Inverted bounds/mask | `if (idx<size) return;` | RFT |
| Truncation | Softmax cut off | raise `max_new_tokens` |
| Const-reassign / syntax | grid-stride `const int idx` | RFT compile reward |

---

## 10 · Repo contents
| Path | What |
|---|---|
| `scripts/sft_kernel.py` | CUDA SFT (PyTorch→CUDA, correct kernels, chat-formatted) |
| `src/densify/kernel_reward.py` | verifiable reward (parse→compile→correct→speedup) + Triton eval, timeout-guarded |
| `scripts/grpo_kernel.py` | GRPO/RLVR (Dr.GRPO + DAPO dynamic sampling + KL anchor) |
| `scripts/eval_worker.py` + `eval_10ops_isolated.py` | **isolated** KernelBench-Lite eval |
| `scripts/head_to_head.py` | ours vs teacher (tok/s + correctness) |
| `scripts/ablate_api_hint.py` / `ablate_triton.py` | prompt ablations (CUDA / Triton) |
| `docs/GRAPHS.md` · `docs/ABLATIONS.md` · `docs/reports/` | all graphs · ablation log · expert report |

## 11 · Next steps
- [ ] **More diverse SFT data** — conv/norm/pooling CUDA examples (addresses 43% of failures)
- [ ] **More GRPO steps** on broader ops — current online GRPO only trains on 3-6 elementwise ops
- [ ] **KernelBench L2** (fusion chains) — where >1x speedups are actually achievable
- [ ] **Teacher model baseline** on KernelBench L1 — in progress
- [ ] **pass@k evaluation** — temperature sampling with k=4 may recover more correct kernels
- [ ] NVFP4 quantization + vLLM serve as a `generate_kernel` tool

## References
RADLADS (arXiv:2505.03005) · Pruning & Distilling MoE into Dense (arXiv:2605.28207) ·
Sakana AI CUDA Engineer / robust-kbench (arXiv:2509.14279) · KernelBench · DeepSeek-R1 GRPO (arXiv:2501.12948) · Dr.GRPO · DAPO.

*Built at the Poolside Laguna XS.2 research hackathon.*
