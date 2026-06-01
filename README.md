# Laguna-Dense — CUDA Kernel Generation

A **~3.0 B fully-dense** model that generates **CUDA / Triton GPU kernels** from PyTorch modules,
**densified from the 33 B [poolside/Laguna-XS.2](https://huggingface.co/poolside/Laguna-XS.2) MoE**.

> Part of the **[laguna-xs2-expert-coactivation-scheduling](https://github.com/cm2435/laguna-xs2-expert-coactivation-scheduling)**
> project (MoE→dense densification). This repo collects **only the CUDA-kernel** work:
> motivation → densification → kernel-mixture pretraining → CUDA SFT → RL → eval harness → results.

---

## Model card — variants & download

All on the Hub under **EvanOLeary** · load with `trust_remote_code=True`.

| Variant | Stage | Size | Repo |
|---|---|---|---|
| **CUDA-SFT** (bf16) | SFT | 5.99 GB | [`EvanOLeary/laguna-xs2-dense-k8-cuda-sft`](https://huggingface.co/EvanOLeary/laguna-xs2-dense-k8-cuda-sft) |
| **CUDA-SFT** · int8 (torchao) | SFT · quant | 3.21 GB (−46%) | [`EvanOLeary/laguna-xs2-dense-k8-cuda-sft-int8`](https://huggingface.co/EvanOLeary/laguna-xs2-dense-k8-cuda-sft-int8) |
| **CUDA-GRPO** | online GRPO | 5.99 GB | [`EvanOLeary/laguna-xs2-dense-k8-cuda-grpo`](https://huggingface.co/EvanOLeary/laguna-xs2-dense-k8-cuda-grpo) |
| **CUDA-DPO** | DPO | 5.99 GB | [`EvanOLeary/laguna-xs2-dense-k8-cuda-dpo`](https://huggingface.co/EvanOLeary/laguna-xs2-dense-k8-cuda-dpo) |

📊 Full graph index: **[docs/GRAPHS.md](docs/GRAPHS.md)** · 📓 Ablation log: **[docs/ABLATIONS.md](docs/ABLATIONS.md)**

---

## 1 · Motivation — MoE expert activation (why densify)

Before collapsing the MoE we measured how many of Laguna's **256 routed experts** actually fire on
**C4** (161,932 tokens, all 39 sparse layers):

| Metric | Value |
|---|---|
| Experts ever used | **256 / 256** (100%) |
| **Effective experts / layer** | **~158** of 256 |
| Mean per-layer coverage | 99.7% |
| Load Gini (concentration) | **0.53** (peaks mid-stack) |

The routed FFN behaves far **denser** than its 256-way capacity → a dense surrogate is viable, and
**K must exceed top-8**. This motivated **K=8 + DO-ACP warm-start**.

**Per-dataset expert activation** (different code domains use different expert pools):

| Dataset | Kind | Eff. experts/layer | Gini | Expert pool / batch |
|---|---|---|---|---|
| magicoder | NL instruction | 183 | 0.425 | 22.5B (72%) |
| swebench_lite | NL problem | 163 | 0.494 | 20.0B (64%) |
| *c4 (baseline)* | *web text* | *158* | *0.528* | *19.4B (62%)* |
| codefeedback | NL query | 151 | 0.519 | 18.5B (59%) |
| opencodeinstruct | NL to Python | 145 | 0.540 | 17.7B (56%) |
| **kernelbook** | **Triton src** | **108** | **0.646** | **13.3B (42%)** |
| **cuda_kernels** | **CUDA src** | **100** | **0.683** | **12.3B (39%)** |

Kernel/CUDA code concentrates onto ~100-108 experts (39-42% of routed weights) vs ~158-183 for
general text (62-72%). This means a kernel-focused batch touches **half the expert weights** of a
general batch -- exactly the overhead that MoE-to-dense collapse removes.

![expert activation](docs/figures/expert_activation.png)

Full analysis: [gist (C4)](https://gist.github.com/Tyronita/fb28e9c31c2b66cccb70fbd939bd1c43) · [gist (per-dataset: KernelBook/CUDA/Magicoder/SWE-bench)](https://gist.github.com/Tyronita/d472e5664dc8291a1dab83f9f3d73fd5) · [gist (C4 detailed visualisations)](https://gist.github.com/Tyronita/cdcb80969d208b83e3f48cddfbbb1422) · `docs/reports/expert-activation-c4.md`.

---

## 2 · Architecture

**2,996,678,656 params (~3.0 B), 5.99 GB bf16.** Each sparse layer's 256-expert MoE → **one dense
SwiGLU FFN** (width K8×512 = 4096) + the kept shared expert. Attention/embeddings/norms copied from teacher.

| Component | Params | Trained? |
|---|---|---|
| `routed_dense` × 39 | **0.98 B** | ✅ reconstruction + SFT + RL |
| attention × 40 (48/8 GQA, 30 SWA + 10 global) | 1.43 B | ❄️ frozen |
| embed + lm_head | 0.41 B | ✅ SFT + RL (lm_head) |
| shared experts × 39 | 0.12 B | ❄️ frozen |
| **Total** | **3.00 B** | |

Hidden 2048 · 40 layers · 262k ctx · 100,352 vocab · SiLU/SwiGLU.

---

## 3 · Training pipeline

```
poolside/Laguna-XS.2 (33B/3B-active MoE, 256 experts)
   │
   │ Stage 0  DO-ACP warm-start (Gram log-det select 8 experts → concat)
   │ Stage 1  teacher-forced reconstruction on KERNEL mixture  → V2 checkpoint
   │ Stage 2  SFT on SakanaAI CUDA (correct kernels only)     → cuda-sft
   │ Stage 3a GRPO-online (Dr.GRPO + live compilation reward)  → cuda-grpo
   ▼ Stage 3b DPO (preference pairs from Sakana traces)        → cuda-dpo
```

### Stage 0 — DO-ACP warm-start
Gram log-det criterion selects the 8 most informative experts per layer → concatenated into one dense
SwiGLU FFN. This gives the reconstruction a much better starting point than random init.

### Stage 1 — Reconstruction (kernel mixture, V2)
Teacher-forced, all-39-layer-parallel reconstruction of each MoE block's output.

| | Value |
|---|---|
| **Loss** | `mean_ℓ( MSE/mean(yℓ²) + 0.05·(1−cos) )` — normalized MSE + cosine alignment |
| **Trainable** | `routed_dense` only (0.98 B) |
| **Optimizer** | Adafactor, lr 2e-4 |
| Steps / tokens | 2000 / ~8.2 M |
| Result | Loss 0.67 → **0.16**, deep-layer MSE 0.20 → **0.018** |
| Hardware | 1× H100, ~35 min |

**Training data (kernel mixture):**

| Dataset | Weight | Language |
|---|---|---|
| `GPUMODE/KernelBook` | 40% | Python→Triton |
| `nvidia/OpenCodeInstruct` | 30% | Python |
| `SakanaAI/AI-CUDA-Engineer-Archive` | 20% | PyTorch→CUDA-C++ |
| Triton multiturn traces | 10% | Triton reasoning |

![v2 reconstruction](docs/figures/v2_training.png)
![v2 per-layer heatmap](docs/figures/v2_layer_heatmap.png)

### Stage 2 — SFT (CUDA kernel generation)

| | Value |
|---|---|
| **Data** | [`SakanaAI/AI-CUDA-Engineer-Archive`](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive) (~30,615 rows, `Correct==True` only) |
| **Format** | chat: `system + user(PyTorch)` → `assistant(CUDA-C++)`, prompt masked |
| **Loss** | Cross-entropy on CUDA completion |
| **Trainable** | `routed_dense` + `lm_head` + norms (1.19 B) |
| **Optimizer** | AdamW 1e-5, grad-clip 1.0, grad-accum 8, seq 2048 |
| Steps / tokens | 400 / ~3.5 M |
| Result | **CE 0.675 → 0.21**; emits working CUDA + restores chat format |

![sft curve](docs/figures/sft_curve.png)

### Stage 3a — GRPO-online (Dr.GRPO + live compilation)

Per prompt, sample 6 kernels → **actually compile + run** each in a subprocess → reward from verification.

| | Value |
|---|---|
| **Algorithm** | Dr.GRPO (unbiased advantage: `r − mean`, no std/length normalization) |
| **Reward** | `+0.1` parse · `+0.2` compile · `+0.4` correct · `+0.3·clip(speedup,0,3)/3` |
| **Sampling** | DAPO dynamic (skip zero-variance groups) |
| **KL anchor** | β=0.02 to SFT reference |
| **Trainable** | `routed_dense` + `lm_head` |
| Steps / group size | 24 / 6 |
| LR / temperature | 1e-6 / 0.9 |
| Tasks | elementwise ops (relu, sigmoid, tanh, gelu, silu, softplus) |

### Stage 3b — DPO (preference pairs)

| | Value |
|---|---|
| **Data** | SakanaAI traces: prefer correct+fastest kernel over incorrect/slow per task |
| **Loss** | DPO (Rafailov et al.): `−log σ(β·Δlogp)` |
| **Reference** | frozen SFT model |
| β / LR | 0.1 / 5e-7 |
| Steps | 300, up to 8 pairs per task, ~200 tasks |

### Training overview
![overview](docs/figures/training_overview.png)

| Stage | Steps | Tokens | Data | Trainable | Loss |
|---|---|---|---|---|---|
| Warm-start (DO-ACP) | — | — | calibration | — | — |
| Reconstruction (V2) | 2000 | ~8.2 M | kernel mixture | routed_dense | MSE+cos |
| SFT (CUDA) | 400 | ~3.5 M | SakanaAI CUDA | +lm_head +norms | CE |
| GRPO-online | 24 | ~144 rollouts | live compilation | routed_dense +lm_head | Dr.GRPO |
| DPO | 300 | ~1600 pairs | Sakana preferences | routed_dense +lm_head | DPO |

---

## 4 · Results — KernelBench Level 1

Full evaluation on [KernelBench](https://github.com/ScalingIntelligence/KernelBench) Level 1
(100 single-operator problems: matmul, conv, activations, norms, pooling, reductions, losses).
Greedy decoding (temperature=0), pass@1, subprocess-isolated evaluation on A100 80GB.

### All variants

| Model | Params | Compile | Correct (fast_0) | Faster (fast_1) | Avg Speedup |
|---|---|---|---|---|---|
| **Teacher** ([Laguna-XS.2](https://huggingface.co/poolside/Laguna-XS.2)) | **33B MoE** | 57% | **24%** | **4%** | 3.1x |
| **GRPO-online** ([`cuda-grpo`](https://huggingface.co/EvanOLeary/laguna-xs2-dense-k8-cuda-grpo)) | 3B dense | 23% | 10% | 1% | 14.6x* |
| **DPO** ([`cuda-dpo`](https://huggingface.co/EvanOLeary/laguna-xs2-dense-k8-cuda-dpo)) | 3B dense | 27% | 2% | 0% | 0.6x |
| **SFT** ([`cuda-sft`](https://huggingface.co/EvanOLeary/laguna-xs2-dense-k8-cuda-sft)) | 3B dense | 27% | 0% | 0% | — |

*Avg speedup inflated by P12 (diagonal matmul, 72–143x algorithmic optimization — legitimate).

### vs frontier models ([KernelBench paper](https://arxiv.org/abs/2502.10517))

| Model | Size | fast_1 (L1) |
|---|---|---|
| DeepSeek R1 | 671B | 12% |
| OpenAI o1 | ~200B | 10% |
| Claude 3.5 Sonnet | ~175B | 10% |
| GPT-4o | ~200B | 4% |
| Llama 3.1-405B | 405B | 3% |
| **Ours (GRPO-online, 3B)** | **3B** | **1%** |

### Key findings

1. **The 33B MoE teacher scores 24% correct** — significantly better than all 3B dense students.
   Densification lost kernel-writing capability that GRPO only partially recovered on the narrow
   set of ops it was trained on. The teacher uniquely solves 17 ops the best student cannot
   (Sigmoid, Softmax, Swish 1.58x, Softsign 2.24x, ELU, HardSigmoid, L1Norm, etc).
2. **RL is essential for the dense student.** SFT compiles (27%) but achieves 0% correctness.
   GRPO reward (compile + correct + speedup) is what teaches the 3B model to produce correct kernels.
3. **The student wins on matmul variants** it was GRPO-trained on, but fails on everything else.
   More diverse SFT/GRPO data is the clear next step.

### Error analysis (GRPO-online vs teacher)

| Error | Teacher (33B) | Student (3B) |
|---|---|---|
| **Correct** | **24** | 10 |
| Compiled but incorrect | 33 | 13 |
| `__init__` missing args | 0 | 43 |
| CUDA build error | 21 | 15 |
| Illegal memory access | 7 | 5 |
| Other | 15 | 14 |

### Eval methodology

**Prompt template** (one-shot, same for all models):
```
You write custom CUDA operators to replace the pytorch operators in the
given architecture to get speedups.

Here's an example to show you the syntax of inline embedding custom CUDA
operators in PyTorch:

Input architecture:
  [elementwise add Model class]

Optimized with CUDA operators:
  [load_inline example with CUDA kernel + ModelNew class]

You are given the following architecture:
  [KernelBench reference code]

Optimize the architecture named Model with custom CUDA operators!
Name your optimized output architecture ModelNew.
```

**Settings**: greedy (temperature=0), max_new_tokens=2048, chat template with `<user>/<assistant>` tags, `</assistant>` as stop token.

**Eval**: KernelBench `eval_kernel_against_ref` -- 5 correctness trials (random inputs, `atol=rtol=1e-4`), 100 timing trials (CUDA events), subprocess-isolated.

### Per-problem results (teacher vs best student)

| PID | Problem | Teacher (33B) | GRPO-online (3B) |
|---|---|---|---|
| 1 | Square matmul | COMPILED | **OK 0.03x** |
| 5 | Matrix scalar mul | **OK 0.65x** | FAIL |
| 7 | Small-K matmul | **OK 0.17x** | COMPILED |
| 8 | Irregular matmul | **OK 0.13x** | **OK 0.13x** |
| 9 | Tall-skinny matmul | **OK 0.18x** | **OK 0.24x** |
| 10 | 3D tensor matmul | COMPILED | **OK 0.12x** |
| 11 | 4D tensor matmul | COMPILED | **OK 0.21x** |
| 12 | Diagonal matmul | **OK 61.2x** | **OK 143.0x** |
| 13 | Symmetric matmul | **OK 0.12x** | COMPILED |
| 14 | Upper-triangular | **OK 1.06x** | COMPILED |
| 15 | Lower-triangular | **OK 0.29x** | COMPILED |
| 17 | Transposed-B matmul | **OK 0.03x** | **OK 0.17x** |
| 19 | ReLU | **OK 0.65x** | **OK 0.91x** |
| 20 | LeakyReLU | **OK 0.65x** | COMPILED |
| 21 | Sigmoid | **OK 0.63x** | FAIL |
| 22 | Tanh | **OK 0.62x** | **OK 0.83x** |
| 23 | Softmax | **OK 0.53x** | FAIL |
| 24 | LogSoftmax | **OK 0.62x** | COMPILED |
| 25 | Swish | **OK 1.58x** | FAIL |
| 26 | GELU | **OK 0.58x** | FAIL |
| 28 | HardSigmoid | **OK 0.64x** | FAIL |
| 29 | Softplus | **OK 0.58x** | FAIL |
| 30 | Softsign | **OK 2.24x** | FAIL |
| 31 | ELU | **OK 0.65x** | FAIL |
| 32 | HardTanh | **OK 0.65x** | COMPILED |
| 38 | L1Norm | **OK** | FAIL |
| 99 | TripletMarginLoss | **OK 0.83x** | FAIL |

**Teacher uniquely correct (17)**: P5, P7, P13-15, P20-21, P23-26, P28-32, P38, P99
**Student uniquely correct (3)**: P1, P10, P11
**Both correct (7)**: P8, P9, P12, P17, P19, P22

### Teacher inference speed

| Model | Params | VRAM | tok/s (A100) |
|---|---|---|---|
| Teacher (Laguna-XS.2 MoE) | 33B total / 3B active | 67 GB | 12.5 |
| Student (dense bf16) | 3B | 6 GB | 15.4 (eager) / 32.9 (compiled) |
| Student (int8 torchao) | 3B | 3.2 GB | ~10 (eager) / ~22 (compiled) |
| Student (vLLM batched x64) | 3B | 6 GB | 1,227 aggregate |

The student is **2.6x faster** single-seq (compiled) and **98x faster** batched vs the teacher -- while fitting in 10x less VRAM.

### About the teacher -- Laguna-XS.2

[poolside/Laguna-XS.2](https://huggingface.co/poolside/Laguna-XS.2) is a **33.4B parameter MoE** code model by [Poolside AI](https://poolside.ai):
- **256 routed experts**, top-8 routing + 1 shared expert per layer -> **3B active** per token
- 40 layers, 2048 hidden, 48/8 GQA heads, 262k context
- Sigmoid routing activation (not softmax), SwiGLU FFN
- 30 sliding-window attention layers + 10 full-attention layers
- Trained on code (details not public), 100,352 vocab with chat template
- The model that our dense student is distilled from

Full per-problem results: [`results/03_kernelbench_l1/`](results/03_kernelbench_l1/)

---

## 5 · Failure taxonomy (from generated CUDA)

| Category | Example | Fix |
|---|---|---|
| Wrong math/formula | GeLU / Sigmoid / Softmax | GRPO correctness reward |
| Deprecated API | `input.type()` vs `.scalar_type()` | prompt hint / GRPO |
| Inverted bounds/mask | `if (idx<size) return;` | GRPO |
| Truncation (hit token limit) | Softmax cut off mid-kernel | raise `max_new_tokens` |
| `__init__` missing args | conv/norm/pool need weights | broader SFT data |

---

## 6 · Reproducibility — isolate kernel evaluation

Running generated CUDA **in the model's process is INVALID**: a buggy kernel (out-of-bounds write)
corrupts the CUDA context and makes **every later eval fail**, regardless of the model →
order-dependent, contaminated results. **Compile + run each kernel in its own subprocess**
(`scripts/eval_worker.py`). Verified: a crashing kernel segfaults only the worker; the driver survives.

---

## 7 · Inference optimisation

### Quantization (verified — A100, torch 2.12 / transformers 5.9)

| Recipe | Size | Quality | Speed |
|---|---|---|---|
| **torchao Int8 weight-only** (recommended) | 5.99 → 3.21 GB | byte-identical on greedy | −34% tok/s (torch.compile recovers most) |
| HQQ 4-bit (nbits=4, group=64, axis=1) | 5.99 → ~1.7 GB | minor drift; valid CUDA | ~5.8 tok/s |

### Inference backends (A100, dense model)

| Backend | TTFT | Single-seq tok/s | Notes |
|---|---|---|---|
| HF transformers, bf16 eager | 71 ms | 15.4 | Baseline |
| HF + `torch.compile` (mode="default") | 44 ms | **32.9** (2.1×) | Best single-seq |
| vLLM 0.22 (dense plugin) | 51 ms | 21.6 | Best for batched/serving |

### vLLM batched throughput (continuous batching)

| Batch | 1 | 8 | 32 | 64 |
|---|---|---|---|---|
| Aggregate tok/s | 21 | 161 | 621 | **1227** |

→ **~80× HF-eager at batch 64**. Real run: 64 kernels generated in 31s.

vLLM's native `laguna.py` is the MoE teacher; the dense student loads via a ~20-line `LagunaDenseFFN`
plugin that reuses `LagunaMLP`. Patch + run command: [`docs/INFERENCE.md`](docs/INFERENCE.md).

**Recommendation:** one-off generation → HF + `torch.compile`. Rollouts / eval / serving → vLLM.

### Sampling settings

| Knob | Value |
|---|---|
| temperature / top_k | 0.6 / 20 |
| max_new_tokens | ≥ 1024 (under-capping truncates kernels) |
| do_sample | True (→ pass@k; same prompt gives different kernels each sample) |
| enable_thinking | False |

---

## 8 · Repo contents

| Path | What |
|---|---|
| `scripts/sft_kernel.py` | CUDA SFT training |
| `scripts/grpo_kernel.py` | GRPO-online (Dr.GRPO + DAPO + live compilation) |
| `scripts/dpo_sakana.py` | DPO on Sakana preference pairs |
| `src/densify/kernel_reward.py` | Verifiable reward (parse→compile→correct→speedup) |
| `scripts/eval_worker.py` | Subprocess-isolated kernel evaluator |
| `scripts/kernelbench_l1_eval.py` | Full KernelBench L1 evaluation pipeline |
| `scripts/kb_pipeline.py` | Fast pipeline (batch gen + parallel eval) |
| `results/` | All evaluation results + analysis |
| `docs/` | Graphs, ablations, inference guide |

## 9 · Next steps

- [ ] **More diverse SFT data** — conv/norm/pooling CUDA examples (addresses 43% of student failures)
- [ ] **More GRPO steps** on broader ops — current GRPO only trains on 3–6 elementwise ops
- [ ] **KernelBench L2** (fusion chains) — where >1x speedups are achievable
- [ ] **pass@k evaluation** — temperature sampling with k=4 may recover more correct kernels

## References & attribution

### Base model
- **Laguna-XS.2** — [poolside/Laguna-XS.2](https://huggingface.co/poolside/Laguna-XS.2) · [Poolside AI](https://poolside.ai). The 33B/3B-active MoE teacher model this work densifies. 256 routed experts, top-8 + shared, SwiGLU, 262k context.

### Densification method
- **RADLADS** — *Routing-Aware Dense Layerwise Approximation for Dense Surrogates* · KRAFTON AI · [arXiv:2505.03005](https://arxiv.org/abs/2505.03005). The DO-ACP warm-start (Gram log-det expert selection) and teacher-forced reconstruction recipe used in Stage 0–1.
- **Pruning & Distilling MoE into Dense** — [arXiv:2605.28207](https://arxiv.org/abs/2605.28207). Motivation and framing for MoE→dense compression via width-scaled FFN replacement.

### Training data
- **SakanaAI AI-CUDA-Engineer-Archive** — [SakanaAI/AI-CUDA-Engineer-Archive](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive) · [Sakana AI](https://sakana.ai) · [arXiv:2509.14279](https://arxiv.org/abs/2509.14279). The ~30k verified PyTorch→CUDA pairs used for SFT (Stage 2), offline GRPO rewards, and DPO preference pairs (Stage 3). Also the source of the `robust-kbench` anti-reward-hacking methodology.
- **KernelBook** — [GPUMODE/KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook) · [GPU MODE](https://gpumode.com). Python→Triton kernel examples (40% of reconstruction mixture).
- **OpenCodeInstruct** — [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) · NVIDIA. General Python code-instruct data (30% of reconstruction mixture).

### RL algorithms
- **GRPO** — *Group Relative Policy Optimization* · DeepSeek · [arXiv:2501.12948](https://arxiv.org/abs/2501.12948). The base RL algorithm: sample G completions per prompt, compute group-relative advantages, policy-gradient update with KL anchor. Used in Stage 3a.
- **Dr.GRPO** — *Don't Repeat GRPO* · [arXiv:2503.20783](https://arxiv.org/abs/2503.20783). Fix to GRPO's advantage normalization: drops std/length division → unbiased advantage `r − mean(r)`. Prevents length-gaming. Used in both our GRPO and DPO arms.
- **DAPO** — *Dynamic Sampling Policy Optimization* · [arXiv:2503.14476](https://arxiv.org/abs/2503.14476). Skip zero-variance groups (no learning signal) → saves expensive nvcc compiles. Used in Stage 3a.
- **RLVR** — *RL with Verifiable Rewards* (Tülu3 / Kimi-k1.5). The principle of using a verifiable function (compile→correct→speedup) rather than a learned reward model. Our Stage 3a reward is fully verifiable.
- **DPO** — *Direct Preference Optimization* · Rafailov et al. · [arXiv:2305.18290](https://arxiv.org/abs/2305.18290). Loss `−log σ(β·Δlogp)` on preference pairs. Used in Stage 3b.

### Evaluation
- **KernelBench** — *Can LLMs Write Efficient GPU Kernels?* · Stanford Scaling Intelligence Lab · [arXiv:2502.10517](https://arxiv.org/abs/2502.10517) · [github](https://github.com/ScalingIntelligence/KernelBench) · [kernelbench.com](https://kernelbench.com). The benchmark framework (L1–L4) and eval methodology (subprocess-isolated, `eval_kernel_against_ref`, CUDA event timing) used throughout.
- **Kevin** — *Cognition + Stanford agentic kernel generation* · [arXiv:2507.11948](https://arxiv.org/abs/2507.11948). Reward hacking taxonomy referenced in our failure analysis.

### Inference
- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm). Serving engine with native `laguna.py` for the MoE teacher; dense student via `LagunaDenseFFN` plugin.
- **torchao** — [pytorch/ao](https://github.com/pytorch/ao). Int8 weight-only quantization (3.21 GB, byte-identical greedy).
- **HQQ** — *Half-Quadratic Quantization* · [mobiusml/hqq](https://github.com/mobiusml/hqq). 4-bit quantization (~1.7 GB).

### Inline attributions in the training pipeline

| Stage | What we used | From |
|---|---|---|
| DO-ACP warm-start | Gram log-det expert selection | RADLADS (KRAFTON) |
| Reconstruction loss | Normalized MSE + cosine | RADLADS (KRAFTON) |
| SFT data | Verified PyTorch→CUDA pairs | SakanaAI |
| GRPO algorithm | Group-relative policy optimization | DeepSeek R1 |
| Dr.GRPO fix | Unbiased advantage (no std normalization) | Dr.GRPO |
| DAPO sampling | Skip zero-variance groups | DAPO |
| Live compilation reward | Subprocess-isolated compile→correct→speedup | SakanaAI robust-kbench |
| DPO loss | `−log σ(β·Δlogp)` on preference pairs | Rafailov et al. |
| Eval framework | KernelBench L1 (100 problems, fp32, A100) | Stanford Scaling Intelligence |

---

*Built at the Poolside Laguna XS.2 research hackathon.*
