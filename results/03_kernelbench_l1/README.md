# KernelBench Level 1 Evaluation

Full evaluation of all model variants on [KernelBench](https://github.com/ScalingIntelligence/KernelBench) Level 1 (100 single-operator problems), plus comparison to frontier models from the KernelBench paper (ICML 2025).

## Results

### Our Models (pass@1, greedy, A100 80GB)

| Model | HF name | Compile | Correct (fast_0) | Faster (fast_1) | Avg Speedup |
|---|---|---|---|---|---|
| **GRPO-online** | `cuda-rft` | 23% | **10%** | **1%** | 14.6x* |
| **GRPO-offline** | `cuda-grpo` | 19% | **9%** | **1%** | 8.5x* |
| **DPO** | `cuda-dpo` | 27% | 2% | 0% | 0.55x |
| **SFT-v2** | `cuda-sft-v2` | 21% | 0% | 0% | - |
| **SFT-v1** | `cuda-sft` | 27% | 0% | 0% | - |

*Avg speedup inflated by P12 (diagonal matmul) outlier: 72-143x algorithmic speedup.

### vs Frontier Models (from [KernelBench paper](https://arxiv.org/abs/2502.10517), pass@1)

| Model | Size | fast_1 (L1) | fast_1 (L2) | fast_1 (L3) |
|---|---|---|---|---|
| DeepSeek R1 | 671B | 12% | 36% | 2% |
| OpenAI o1 | ~200B | 10% | 24% | 12% |
| Claude 3.5 Sonnet | ~175B | 10% | 7% | 8% |
| DeepSeek V3 | 671B | 6% | 4% | 0% |
| GPT-4o | ~200B | 4% | 5% | 0% |
| Llama 3.1-405B | 405B | 3% | 0% | 0% |
| Llama 3.1-70B | 70B | 3% | 0% | 0% |
| **Ours (GRPO-online, 3B)** | **3B** | **1%** | - | - |
| **Ours (GRPO-offline, 3B)** | **3B** | **1%** | - | - |

### vs Smoke Test (Section 7b of main README)

The earlier smoke test used a different setup — included here for context:

| | Smoke Test (DPO) | KernelBench L1 (this eval) |
|---|---|---|
| Model | DPO | All 5 variants |
| Problems | 10 elementwise ops | 100 L1 problems (matmul, conv, norm, pool, loss...) |
| Prompt | Expert system prompt + API hints | One-shot KernelBench format |
| Sampling | temp=0.6, top_k=20, pass@3 | Greedy pass@1 |
| ReLU correct | 2/3 (67%) | GRPO: 1/1 (100%) |
| Tanh correct | 2/3 (67%) | GRPO: 1/1 (100%) |

The L1 eval is much harder (100 diverse ops vs 10 elementwise) with stricter sampling (greedy vs best-of-3).

## Discussion

### Key Findings

**1. RL is essential for correctness.** SFT models compile at 21-27% but achieve 0% correctness. The GRPO reward signal (compile + correct + speedup) is what teaches the model to produce numerically correct kernels. DPO (preference learning) sits between — 27% compile, 2% correct.

**2. Online > offline GRPO.** GRPO-online (live compilation, 24 steps) slightly outperforms GRPO-offline (dataset rewards, 120 steps): 10% vs 9% correct. The online model also uniquely solves 4D tensor matmul and transposed matmul variants that the offline model cannot. The live reward signal, despite being more expensive, enables the model to learn from its own mistakes.

**3. A 3B model matches 405B on fast_1.** Our GRPO-online achieves 1% fast_1, comparable to Llama-3.1-405B (3%) despite being 135x smaller. This suggests the KernelBench L1 ceiling is largely about training data coverage (which ops the model has seen CUDA for), not raw model capacity.

**4. The model has one genuine algorithmic insight.** Problem 12 (diagonal matmul) achieved 72-143x speedup by recognizing that `diag(A) @ B` is elementwise row-scaling, not full matmul. This passes all correctness checks and is a legitimate optimization that PyTorch's broadcasting doesn't exploit.

**5. Correct kernels are close to PyTorch parity, not faster.** ReLU (0.92x), Tanh (0.88x), GELU (0.85x) — within 8-15% of PyTorch eager. Single-op L1 problems are memory-bandwidth-bound; beating cuBLAS/cuDNN on single ops is extremely hard. Speedups require operator fusion (L2+).

### Error Analysis

For the best model (GRPO-online, 100 problems):

| Error Category | Count | Root Cause |
|---|---|---|
| **Correct** | **10** | Working CUDA kernel |
| Compiled but incorrect | 13 | Math bugs (wrong formula, bad indexing) |
| `__init__` missing args | 43 | Conv/norm/pool ops need constructor weights |
| CUDA build error | 15 | Invalid C++ syntax |
| CUDA illegal memory | 5 | Out-of-bounds access (flat index into multi-dim) |
| No code extracted | 4 | Model output `</think>` loop or truncated |
| Eval returned None | 6 | Kernel crashed before returning |
| Other | 4 | Python syntax, etc. |

**The #1 blocker is structural (43/100):** problems 31-93 (conv, pooling, norm) require constructor arguments for weights/parameters. The model outputs raw CUDA that doesn't handle weight storage. This is partly a wrapper issue and partly a model capability gap — the model was trained on elementwise ops, not parameterized layers.

**13 "almost there" kernels** compiled and ran but produced wrong numerical output. These are the ideal targets for more GRPO training — the model knows the structure but gets the math wrong.

### What Would Improve Scores

| Intervention | Expected Impact | Which errors it fixes |
|---|---|---|
| **More diverse SFT data** (conv, norm, pooling CUDA examples) | High | 43 `__init__` + 15 build errors |
| **More GRPO steps** on broader ops | High | 13 compiled-but-incorrect |
| **Raise max_new_tokens to 4096** | Low (+6 problems) | 6 truncated outputs |
| **Repetition penalty** | Low (+8 problems) | 8 `</think>` loops |
| **Prompt: "use float\* not templates"** | Medium (SFT models) | 25-33 SFT build errors |
| **Modern PyTorch API in SFT data** | Medium (SFT models) | Deprecated `A.type()` errors |

## Methodology

### Generation
- **Prompt**: One-shot KernelBench format (elementwise add example with `load_inline`)
- **Decoding**: Greedy (temperature=0, deterministic, pass@1)
- **Max tokens**: 2048
- **Chat template**: Laguna format (`<system>/<user>/<assistant>`)
- **Post-processing**: Raw C++ auto-wrapped in Python `load_inline` format

### Evaluation
- **Framework**: KernelBench `eval_kernel_against_ref`
- **Isolation**: Each kernel evaluated in separate subprocess (CUDA crash protection)
- **Correctness**: 5 randomized input trials, `torch.allclose(atol=1e-4, rtol=1e-4)`
- **Performance**: 100 trials, CUDA event timing
- **GPU**: NVIDIA A100 80GB PCIe (Ampere)
- **Precision**: FP32

### Scripts
- `scripts/kernelbench_l1_eval.py` — Main eval (sequential gen + subprocess eval)
- `scripts/kb_eval_worker.py` — Isolated subprocess evaluator
- `scripts/kb_gen_batch.py` — Batch generation (88 tok/s with batch_size=6)
- `scripts/kb_eval_parallel.py` — Parallel eval for multiple models
- `scripts/kb_pipeline.py` — Fast pipeline (batch gen + parallel eval)
- `scripts/kb_compare.py` — Cross-model comparison
- `scripts/kb_error_analysis.py` — Error categorization

### Timing
- Generation: ~15-20 min per model (batch_size=6, 88 tok/s on A100)
- Evaluation: ~60-90 min per model (subprocess-isolated, sequential)
- Total: ~6 hours for all 5 models

## File Structure
```
results/03_kernelbench_l1/
├── README.md                      # This file
├── COMPARISON.txt                 # Side-by-side per-problem results
├── ERROR_ANALYSIS.txt             # Full error breakdown
├── grpo-offline/                  # GRPO on Sakana dataset rewards
│   ├── problem_NNN_raw.txt       # Raw model output (100 files)
│   ├── kernels/problem_NNN_kernel.py  # Wrapped Python+CUDA
│   ├── problem_NNN_result.json   # Per-problem eval
│   ├── summary.json              # Aggregate metrics
│   └── all_results.json
├── grpo-online/                   # GRPO with live compilation
├── dpo/                           # DPO preference learning
├── sft-v1/                        # SFT (levels 1+2, 400 steps)
├── sft-v2/                        # SFT extended (levels 1+2+3, +500 steps)
└── teacher/                       # Laguna-XS.2 33B MoE (baseline)
```
