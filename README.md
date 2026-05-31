# Laguna-Dense — CUDA Kernel Generation

> **Group research — shared learnings.** This repo and its docs capture our group's
> work densifying the Laguna-XS.2 MoE into a dense CUDA-kernel model. See
> [`docs/TRAINING_PROVENANCE.md`](docs/TRAINING_PROVENANCE.md) and
> [`docs/INVESTIGATION_GENERAL_METHOD.md`](docs/INVESTIGATION_GENERAL_METHOD.md).


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
| **Method** | DO-ACP warm-start → teacher-forced reconstruction (kernel mix) → CUDA SFT → [GRPO next] |
| **vs teacher** | **11× fewer params, 12× less VRAM, +26 % faster decode** (32.1 vs 25.4 tok/s) |
| **Status** | SFT done; emits correct CUDA on simple ops (ReLU/Tanh); GRPO next |

## Pipeline / lineage
```
poolside/Laguna-XS.2 (33B/3B-active MoE, 256 experts)
   │ densify: routed experts → 1 dense SwiGLU (K=8) per layer
   │ Stage 0  DO-ACP warm-start (Gram log-det select 8 experts → concat)
   │ Stage 1  teacher-forced reconstruction on a KERNEL mixture  → "V2" checkpoint
   │ Stage 2  SFT on SakanaAI CUDA                                → THIS MODEL
   ▼ Stage 3  GRPO/RLVR (verifiable reward)                      → [next]
```

## Model card — variants & download
All on the Hub under **[`EvanOLeary`](https://huggingface.co/EvanOLeary)** · load with `trust_remote_code=True`.

| Variant | Stage | Size | Repo |
|---|---|---|---|
| **CUDA-SFT** (bf16) | SFT | 5.99 GB | [`…-cuda-sft`](https://huggingface.co/EvanOLeary/laguna-xs2-dense-k8-cuda-sft) |
| CUDA-SFT · **int8** (torchao) | SFT · quant | **3.21 GB (−46 %)** | [`…-cuda-sft-int8`](https://huggingface.co/EvanOLeary/laguna-xs2-dense-k8-cuda-sft-int8) |
| CUDA-SFT · **4-bit HQQ** | SFT · quant | **~1.7 GB (−72 %)** | [`…-cuda-sft-int4-hqq`](https://huggingface.co/EvanOLeary/laguna-xs2-dense-k8-cuda-sft-int4-hqq) |
| CUDA-**GRPO** | online GRPO | 5.99 GB | [`…-cuda-grpo`](https://huggingface.co/EvanOLeary/laguna-xs2-dense-k8-cuda-grpo) |
| CUDA-**DPO** | DPO | 5.99 GB | [`…-cuda-dpo`](https://huggingface.co/EvanOLeary/laguna-xs2-dense-k8-cuda-dpo) |
| Recon · kernel-mix (V2) | pretrain | 5.99 GB | [`…-k8-kernelmix`](https://huggingface.co/EvanOLeary/laguna-xs2-dense-k8-kernelmix) |
| Recon · Python (V1) | pretrain | 5.99 GB | [`…-k8-recon`](https://huggingface.co/EvanOLeary/laguna-xs2-dense-k8-recon) |

### Quantization (verified — A100, torch 2.12 / transformers 5.9)
| Recipe | Size | Quality | Speed | How |
|---|---|---|---|---|
| **torchao Int8 weight-only** *(recommended)* | 5.99 → **3.21 GB** | **byte-identical** on greedy | −34 % tok/s (`torch.compile` recovers most) | `quantize_(model, Int8WeightOnlyConfig())` (~0.4 s, save `.bin`) |
| **HQQ 4-bit** (`nbits=4, group=64, axis=1`) | 5.99 → **~1.7 GB** | minor drift; valid CUDA | ~5.8 tok/s | `AutoHQQHFModel.quantize_model(...)` — no calibration |
| ❌ bitsandbytes 0.49 · ❌ torchao Int4 (needs `mslk`) · ❌ NVFP4 (Blackwell) · ❌ FP8 (Hopper) | — | — | — | unsupported on Ampere/this stack |

### Inference — measured (A100, dense model)
| Backend | TTFT | single-seq | batched throughput | output |
|---|---|---|---|---|
| HF transformers, bf16 eager | 71 ms | 15.4 tok/s | — | ✅ valid CUDA |
| HF + **`torch.compile`** (`mode="default"`) | **44 ms** | **32.9 tok/s (2.1×)** | — | ✅ |
| **vLLM 0.22 (dense plugin)** | 51 ms | 21.6 tok/s | **see below** | ✅ valid CUDA |

**vLLM batched throughput** (continuous batching — the win for GRPO rollouts/serving):

| batch | 1 | 8 | 32 | 64 |
|---|---|---|---|---|
| aggregate tok/s | 21 | 161 | 621 | **1227** |

→ **~80× HF-eager** at batch 64 (per-request steady ~19 tok/s). Real run: **64 kernels generated in 31 s**.

### ✅ vLLM serving works (dense student)
vLLM's native `laguna.py` is the **MoE teacher**; the dense student loads via a **~20-line,
`model_type`-gated `LagunaDenseFFN`** that reuses native `LagunaMLP` (so **OG Laguna is untouched** —
safe to upstream). Patch + run command + gotchas (`apply_chat_template`, `VLLM_USE_FLASHINFER_SAMPLER=0`):
**[`docs/INFERENCE.md`](docs/INFERENCE.md)** · diff: [`docs/vllm_laguna_dense.patch`](docs/vllm_laguna_dense.patch) ·
repro: `scripts/bench_vllm_dense.py`, `scripts/bench_vllm_batch.py`.

- **One-off generation** → HF + `torch.compile` (fastest single-seq). **Rollouts / eval / serving** → vLLM (~1227 tok/s).
- **Sampling:** `temperature 0.6 · top_k 20` → **pass@k**; `max_new_tokens ≥ 1024` (under-capping truncates kernels). **On-device:** ExecuTorch (fits mobile at 4-bit).
- **Eval isolation:** always compile+run generated kernels in a **subprocess** (`scripts/eval_worker.py`) — a faulty kernel corrupts the CUDA context otherwise.

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
| Not used | `CUDA_Speedup_Native`, `NCU_Profile`, `Clang_Tidy` → reserved for the GRPO reward |

### Datasets — links & contents
| Dataset | Link | Contents |
|---|---|---|
| **GPUMODE/KernelBook** | [🤗](https://huggingface.co/datasets/GPUMODE/KernelBook) | PyTorch→**Triton** kernel pairs (`python_code` → `triton_code`) scraped + compiled; the kernel-generation backbone of the mix |
| **nvidia/OpenCodeInstruct** | [🤗](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | ~5 M general **Python** instruction→code pairs; anti-monoculture / keeps general coding ability |
| **SakanaAI/AI-CUDA-Engineer-Archive** | [🤗](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive) | **PyTorch→CUDA-C++** kernels discovered by Sakana's agent; `level_1/2/3` splits, per-row `Correct`, `CUDA_Speedup_Native`, `NCU_Profile`, `Clang_Tidy`; ~30,615 `Correct==True` rows used for SFT/GRPO/DPO (CC-BY-4.0) |
| **kernelbook-triton multiturn traces** | [🤗](https://huggingface.co/datasets/ppbhatt500/kernelbook-triton-multiturn-reasoning-traces) | multi-turn **Triton reasoning** traces (think→kernel); adds reasoning-shaped kernel data |

**Eval substrate:** [KernelBench](https://github.com/ScalingIntelligence/KernelBench) (L1 single ops · L2 fusion · L3 nets · L4 HF-model) + the isolated `robust-kbench`-style reward.

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

### 7a · Speed & size vs teacher (head-to-head, same 6 CUDA questions) — **VALID**
| | OURS (3.0 B dense) | TEACHER Laguna-XS.2 |
|---|---|---|
| Params | **3.0 B** | 33.4 B |
| VRAM / load | **6 GB / 3 s** | 67 GB / 35 s |
| **Decode speed** | **32.1 tok/s** | 25.4 tok/s |
→ **11× smaller, 12× less VRAM, +26 % faster.** Neither model beats PyTorch eager on single
elementwise ops (memory-bandwidth-bound — speedups need fusion / KernelBench L2).

### 7b · Correctness — simple ops (cross-validated, subprocess-isolated) — **VALID**
Reliable on simple elementwise ops; consistent across **three independent harnesses** (best-of-4, isolated pass@3, per-kernel re-eval):

| Op | pass@4 (best-of-4) | pass@3 (isolated) | speedup vs eager |
|---|---|---|---|
| **ReLU** | **3/4 correct** | 2/3 correct | **0.93×** |
| **Tanh** | **3/4 correct** | 2/3 correct | — |
| Sigmoid | 0/4 | 0/3 | fails |

**Read:** ReLU & Tanh land ~70–75 % correct at pass@k (three runs agree → trustworthy). Harder ops
(Sigmoid/GeLU) consistently fail — the model botches **float4-vectorization casts**
(`float4* v = float4* x;` instead of `reinterpret_cast<float4*>(x)`) → the GRPO **compile reward** target.
No single elementwise op beats eager (memory-bandwidth-bound; ReLU 0.93×) — speedups need fusion (L2).

### What's in KernelBench (the benchmark)
| Level | # | Contents |
|---|---|---|
| L1 | 100 | single ops — mostly **matmul/conv** + activations/norms/reductions/losses |
| L2 | 100 | **fusion** chains (where >1× speedup is winnable) |
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
| Wrong math/formula | GeLU / Sigmoid / Softmax | GRPO correctness reward |
| Deprecated API | `input.type()` vs `.scalar_type()` | prompt hint / GRPO |
| Inverted bounds/mask | `if (idx<size) return;` | GRPO |
| Truncation | Softmax cut off | raise `max_new_tokens` |
| Const-reassign / syntax | grid-stride `const int idx` | GRPO compile reward |

---

## 10 · Repo layout — the `000–004` pipeline (CUDA post-training separated)

The numbered pipeline reads top-to-bottom. **Stages 0–1 are the densification core** (shared with
the [cm2435 research repo](https://github.com/cm2435/laguna-xs2-expert-coactivation-scheduling));
**stages 2–4 + reward + eval are the CUDA-focused post-training** that is the point of *this* repo.

**① Densification core** (MoE → dense)
| Script | Stage | Trains |
|---|---|---|
| `scripts/000_build_dense_placeholder.py` | build + DO-ACP warm-start (`--init {random,selected-concat}`) | — |
| `scripts/001_train_dense_reconstruction.py` | teacher-forced reconstruction | `routed_dense` |
| `src/densify/{densify_layer,reconstruction,dense_checkpoint/*}.py` | DO-ACP + reconstruction + dense-model defn | — |

**② CUDA post-training** ⭐ *(the CUDA-focused work)*
| Script | Stage | Trains |
|---|---|---|
| `scripts/002_sft_general.py` · `scripts/002_sft_cuda.py` | SFT (general / Sakana CUDA) | `routed_dense + lm_head + norms` |
| `scripts/003_grpo.py` | GRPO/RLVR (Dr.GRPO + DAPO; **isolated-parallel reward**) | `routed_dense + lm_head` |
| `scripts/003_grpo_offline.py` | offline GRPO on Sakana traces | `routed_dense + lm_head` |
| `scripts/004_dpo.py` | DPO (correct+fast ≻ incorrect/slow) | `routed_dense + lm_head` |
| `src/densify/kernel_reward.py` | verifiable reward (parse→compile→correct→speedup) + **`reward_for_text_isolated`** (subprocess) | — |

**③ CUDA eval & ablations**
| Script | What |
|---|---|
| `scripts/eval_worker.py` · `scripts/kernelbench_lite_eval.py` · `eval_10ops_isolated.py` | **isolated** KernelBench-Lite eval (subprocess per kernel) |
| `scripts/head_to_head.py` | ours vs teacher (tok/s + correctness) |
| `scripts/ablate_api_hint.py` · `ablate_triton.py` | prompt ablations (CUDA / Triton) |

**Docs** · [`TRAINING_PROVENANCE`](docs/TRAINING_PROVENANCE.md) (per-stage trainable params) · [`INVESTIGATION_GENERAL_METHOD`](docs/INVESTIGATION_GENERAL_METHOD.md) (random vs lift-and-shift, confirmed) · [`REPRODUCE`](docs/REPRODUCE.md) (GRPO/DPO deep guides) · [`PROVENANCE`](docs/PROVENANCE.md) · [`GRAPHS`](docs/GRAPHS.md) · [`ABLATIONS`](docs/ABLATIONS.md) · **[consolidation/PR plan →](docs/PR_PLAN.md)**

## 11 · Next — GRPO (GRPO/RLVR)
Sample G kernels/prompt → reward = **compile + correct + speedup** (via `robust-kbench`) → Dr.GRPO
advantage + DAPO dynamic sampling + KL-to-SFT anchor → optimize **KernelBench `fast_1`** → NVFP4 +
vLLM serve as a `generate_kernel` tool.

## References
RADLADS (arXiv:2505.03005) · Pruning & Distilling MoE into Dense (arXiv:2605.28207) ·
Sakana AI CUDA Engineer / robust-kbench (arXiv:2509.14279) · KernelBench · DeepSeek-R1 GRPO (arXiv:2501.12948) · Dr.GRPO · DAPO.

*Built at the Poolside Laguna XS.2 research hackathon.*

## Reproduction & provenance
- **[docs/REPRODUCE.md](docs/REPRODUCE.md)** — full end-to-end reproduction with checkpointing, plus deep guides for **GRPO** and **DPO**.
- **[docs/PROVENANCE.md](docs/PROVENANCE.md)** — who-introduced-what (verified from commits) with code samples and paper provenance (incl. DO-ACP / KRAFTON / RADLADS).
- **[training/](training/)** — the ported end-to-end training scripts + data mixtures.
