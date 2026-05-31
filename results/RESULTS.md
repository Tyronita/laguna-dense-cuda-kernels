# Results — Post-training arms & prompt ablation (Laguna-Dense CUDA, ~3B)

All post-training arms start from **SFT-extended** (`EvanOLeary/laguna-xs2-dense-k8-cuda-sft-v2`).
Eval = generate → **subprocess-isolated** compile → correctness vs PyTorch eager.

## Prompt ablation — what the CUDA-version pin fixes (SFT-extended, 6 ops, K=4)
| System prompt | compile@4 | correct@4 |
|---|---|---|
| BARE (minimal) | 5/6 | 2/6 |
| **VERSION-only** (`PyTorch 2.7 / CUDA 12.8`) | **6/6** | 2/6 |
| MASTER (all rules) | 2/6 | 0/6 |

**Findings:**
- **The version pin is the winner for compilability** (5/6 → 6/6): steers off deprecated APIs.
- **The long "master" rules prompt HURT the 3B model** (compile 2/6, correct 0/6). Over-prescription
  makes a small model attempt complex constructs it can't get right.
- Version pin did **not** lift *correctness* (2/6 both) — correctness bugs are logic errors, not API issues.

## Post-training arms

Three arms, all from SFT-extended, using different RL methods:

| Arm | HF name | Script | Method | Data | Online? |
|---|---|---|---|---|---|
| **GRPO-offline** | `cuda-grpo` | `rft_offline_sakana.py` | Dr.GRPO + DAPO | SakanaAI traces grouped by Task_ID (~120 tasks) | **No** — uses pre-recorded `Correct` + `CUDA_Speedup_Native` |
| **GRPO-online** | `cuda-rft` | `03_grpo.py` | Dr.GRPO + DAPO | Model's own generations on elementwise ops | **Yes** — live compilation + verification in subprocess |
| **DPO** | `cuda-dpo` | `04_dpo.py` | DPO (Rafailov et al.) | SakanaAI preference pairs (correct+fast ≻ incorrect/slow) | **No** — preferences from dataset |

### GRPO-offline (`cuda-grpo`)
- Groups ~6 candidate kernels per Task_ID from SakanaAI archive
- Reward = `CUDA_Speedup_Native` if `Correct` else 0 (pre-recorded, no compilation)
- Dr.GRPO advantage: `r − mean(r)` (no std/length normalization)
- DAPO: skip zero-variance groups
- KL anchor to SFT-extended (β=0.02)
- 120 steps, lr=1e-6
- **Limitation**: off-policy — sharpens toward Sakana's best traces but cannot exceed the dataset
- **Issue found**: outlier speedups (≤190×) caused loss spikes → clip recommended

### GRPO-online (`cuda-rft`)
- Generates kernels on-policy from the model itself
- Each kernel is actually compiled + run in a subprocess
- Reward: +0.1 parse, +0.2 compile, +0.4 correct, +0.3·clip(speedup,0,3)/3
- Same Dr.GRPO + DAPO + KL anchor as offline
- 24 steps, lr=1e-6, temperature=0.9, group_size=6
- Tasks: elementwise ops (relu, sigmoid, tanh, gelu, silu, softplus)
- **Advantage**: can discover novel improvements beyond the dataset
- **Cost**: much slower (compiles CUDA each step)

### DPO (`cuda-dpo`)
- Per task: prefer correct+fastest kernel over incorrect/slow kernel
- Preference pairs extracted from SakanaAI evolutionary refinement trajectory
- β=0.1, lr=5e-7, 300 steps
- Up to 8 pairs per task, ~200 tasks
- Reference model: frozen SFT-extended (implicit KL anchor)

## Why the HF names are confusing

| HF repo name | What it actually is |
|---|---|
| `cuda-grpo` | Offline GRPO on Sakana dataset rewards (not live GRPO) |
| `cuda-rft` | Online GRPO with live compilation (not rejection fine-tuning) |
| `cuda-dpo` | DPO (this one's correct) |

The naming arose historically: "RFT" was initially planned as rejection sampling, but the implementation evolved into online GRPO with verifiable reward. The "GRPO" model was the second attempt that used the Sakana dataset directly instead of live compilation.

## Models
| Stage | HF | Training |
|---|---|---|
| SFT (level_1+2, 400 steps) | `laguna-xs2-dense-k8-cuda-sft` | CE on correct CUDA pairs |
| SFT-extended (level_1+2+3, +500 steps) | `laguna-xs2-dense-k8-cuda-sft-v2` | Continued CE, broader data |
| GRPO-offline (Sakana traces) | `laguna-xs2-dense-k8-cuda-grpo` | Off-policy Dr.GRPO |
| GRPO-online (live compilation) | `laguna-xs2-dense-k8-cuda-rft` | On-policy Dr.GRPO + RLVR |
| DPO (Sakana preferences) | `laguna-xs2-dense-k8-cuda-dpo` | Preference optimization |
