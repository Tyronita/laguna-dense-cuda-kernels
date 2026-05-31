# KernelBench Level 1 Evaluation

Full evaluation of post-training variants on [KernelBench](https://github.com/ScalingIntelligence/KernelBench) Level 1 (100 problems).

## Models Evaluated

| Model | Training | HuggingFace |
|---|---|---|
| **GRPO** | Dr.GRPO + DAPO dynamic sampling | `EvanOLeary/laguna-xs2-dense-k8-cuda-grpo` |
| **RFT** | Rejection fine-tuning (offline) | `EvanOLeary/laguna-xs2-dense-k8-cuda-rft` |
| **DPO** | Direct preference optimization | `EvanOLeary/laguna-xs2-dense-k8-cuda-dpo` |

All models share the same base: `laguna-xs2-dense-k8-cuda-sft` (3.0B dense, from Laguna XS.2 MoE densification).

## Methodology

### Generation
- **Prompt**: KernelBench standard one-shot format (see `scripts/kernelbench_l1_eval.py`)
  - System: uses model's default chat template (Laguna chat format with `<system>/<user>/<assistant>` tags)
  - User prompt: problem statement + one-shot example (elementwise add with `load_inline`) + target architecture
  - Instruction: "Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew."
- **Decoding**: Greedy (temperature=0, deterministic)
- **Max tokens**: 2048
- **Hardware**: NVIDIA A100 80GB PCIe
- **Post-processing**: Raw C++ outputs auto-wrapped in Python `load_inline` format for KernelBench compatibility

### Evaluation (subprocess-isolated)
Each kernel is evaluated in a separate Python subprocess to prevent CUDA context corruption from buggy kernels (see Section 8 of main README).

- **Correctness**: 5 randomized input trials, allclose comparison vs PyTorch reference
- **Performance**: 100 trials with CUDA event timing
- **Static checks**: KernelBench regex-based reward hacking detector
- **GPU**: NVIDIA A100 80GB PCIe (Ampere)
- **Precision**: FP32

### Scoring
```
Composite = 10% × compile_rate + 40% × correct_rate + 30% × speedup_norm + 20% × faster_rate
```
Where `speedup_norm = min(avg_speedup / 2.0, 1.0)` caps at 2x.

## Existing Smoke Test Results (from DPO model)

The results in Section 7b of the main README ("Correctness — simple ops") were generated using the **DPO model** (`EvanOLeary/laguna-xs2-dense-k8-cuda-dpo`) on 10 elementwise operations with:
- System prompt: explicit CUDA engineering instructions with API hints
- pass@3 sampling (temperature=0.6, top_k=20)
- Subprocess-isolated evaluation via `scripts/eval_10ops_isolated.py`

Those results show ReLU/Tanh at ~70% correct, while harder ops (Sigmoid/GeLU) fail due to float4 vectorization bugs.

## Prompt Comparison

| | Smoke Test (Section 7b) | KernelBench L1 (this eval) |
|---|---|---|
| **Model** | DPO | GRPO / RFT / DPO |
| **Prompt style** | System prompt with API hints | One-shot example (standard KB format) |
| **System prompt** | "Expert GPU kernel engineer. Use scalar_type(), AT_DISPATCH..." | Default model chat system |
| **Sampling** | do_sample=True, temp=0.6, top_k=20, pass@3 | Greedy (temp=0), pass@1 |
| **Problems** | 10 unary elementwise ops | 100 KernelBench L1 (matmul, conv, activations, norms, losses) |
| **Eval** | Custom `kernel_reward.py` | KernelBench `eval_kernel_against_ref` |

## File Structure
```
results/kernelbench_l1/
├── grpo/                          # GRPO model results
│   ├── problem_NNN_raw.txt       # Raw model output (100 files)
│   ├── kernels/                  # Wrapped Python+CUDA kernels
│   │   └── problem_NNN_kernel.py
│   ├── problem_NNN_result.json   # Per-problem eval results
│   ├── summary.json              # Aggregate metrics
│   └── all_results.json          # All results in one file
├── rft/                           # RFT model results (same structure)
├── dpo/                           # DPO model results (same structure)
└── comparison.json                # Side-by-side final comparison
```

## Running the Evaluation

```bash
# Requires: NVIDIA GPU with CUDA 12+, ~80GB VRAM (for generation + eval headroom)
# Install deps
pip install torch transformers

# Install KernelBench
pip install -e /path/to/KernelBench

# Run
export CUDA_HOME=/usr/local/cuda
python scripts/kernelbench_l1_eval.py
```

The script:
1. Loads each model sequentially
2. Generates kernels for all 100 L1 problems (greedy decoding)
3. Evaluates each kernel in a subprocess (CUDA crash isolation)
4. Produces per-problem results + aggregate summary

## Generation Speed
- ~15 tok/s on A100 80GB (custom Laguna architecture, bf16)
- ~30-60s per problem generation
- ~50 min generation per model, ~60 min evaluation per model
- Total: ~5-6 hours for all 3 models
