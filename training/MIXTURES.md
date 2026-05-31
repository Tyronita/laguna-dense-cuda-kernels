# Training data mixtures

Every dataset ID, weight, and field mapping used across the pipeline. The reconstruction
signal depends **only on what text drives the frozen teacher's forward pass**, so the
mixture is a direct quality lever (it sets which experts get activated and therefore which
must be reconstructed).

## Stage 1 — Reconstruction mixtures

The trainer (`scripts/001_train_dense_reconstruction.py`) interleaves streaming datasets by
weight with a **seeded RNG** (`random.Random(0)`, deterministic) via `--datasets
"name:weight[:split]"`. Two mixtures were run:

### Recon-V1 — OpenCode only (baseline)
```
--dataset nvidia/OpenCodeInstruct
```
Result: total loss **0.691 → 0.332**; deep layers (L28–39) **0.232 → 0.025**.

### Recon-V2 — kernel-anchored mixture (≈2× lower final loss)
```
--datasets "GPUMODE/KernelBook:0.40,nvidia/OpenCodeInstruct:0.30,SakanaAI/AI-CUDA-Engineer-Archive:0.20:level_1,ppbhatt500/kernelbook-triton-multiturn-reasoning-traces:0.10"
```
| Source | Weight | Role |
|---|---|---|
| `GPUMODE/KernelBook` | 0.40 | PyTorch → Triton pairs |
| `nvidia/OpenCodeInstruct` | 0.30 | general Python |
| `SakanaAI/AI-CUDA-Engineer-Archive` (split `level_1`) | 0.20 | PyTorch → CUDA-C++ |
| Triton multiturn reasoning traces | 0.10 | Triton reasoning |

≈ 50% kernel / 30% Python / 20% CUDA-C++. Result: loss **0.672 → 0.163**, deep MSE
**0.204 → 0.018**. Kernel inputs route through a narrower, more-reconstructible expert
subset, so the kernel-anchored mix reconstructs better than OpenCode-only.

> Per-dataset split syntax: `name:weight:split` (e.g. Sakana needs `level_1`, **not** `train`).
> Field handling is in `src/densify/reconstruction_data.py::format_sft_row`, which recognizes
> OpenCodeInstruct (`instruction`/`output`), KernelBook/CUDA (`query` + `kernel`/`code`), and
> CodeFeedback schemas without per-dataset glue.

## Stage 3 — the two SFT mixes

### SFT Mix A — general-code recovery
`scripts/002_sft_general.py` on **`nvidia/OpenCodeInstruct`** (rendered to a JSONL of chat
rows first). Recovers chat behaviour + broad code generation after reconstruction. Optional
logit-KD against the teacher via `--kd-dataset/--kd-weight/--kd-temperature`.
- Defaults: seq 8192, lr 5e-5, max-steps 500; trainable adds `--train-norms --train-lm-head`.
- Observed: clean documented Python recovered; HumanEval plateaus ~1/10 (logic, not syntax) →
  motivates RFT/GRPO.

### SFT Mix B — CUDA kernels
`scripts/002_sft_cuda.py` on **`SakanaAI/AI-CUDA-Engineer-Archive`**, splits `level_1,level_2`,
**`Correct==True` only**, field map `PyTorch_Code_Module → CUDA_Code`, chat-formatted.
- Defaults: seq 2048, lr 1e-5, grad-accum 8, max-steps 400; trainable = `routed_dense +
  lm_head + norms` (attention frozen).
- Held out for the RFT reward: `CUDA_Speedup_Native`, `NCU_Profile`, `Clang_Tidy`.

## Stage 5 — DPO preference pairs
`scripts/004_dpo.py` mines the Sakana archive's evolutionary trajectory: per `Task_ID`,
**prefer the correct + fastest kernel (by `CUDA_Speedup_Native`) over an incorrect/slower one**.
Defaults: splits `level_1,level_2`, max-tasks 200, pairs-per-task 8, β 0.1, lr 5e-7, 300 steps.
