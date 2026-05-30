# Results — RFT arms & prompt ablation (Laguna-Dense CUDA, ~3B)

All RFT arms trained from **SFT-extended** (`EvanOLeary/laguna-xs2-dense-k8-cuda-sft-v2`),
offline on the **SakanaAI CUDA Engineer Archive** verified traces (reward = `Correct` +
`CUDA_Speedup_Native`). Eval = generate → **subprocess-isolated** compile → correctness vs PyTorch eager.

## Prompt ablation — what the CUDA-version pin fixes (SFT-extended, 6 ops, K=4)
| System prompt | compile@4 | correct@4 |
|---|---|---|
| BARE (minimal) | 5/6 | 2/6 |
| **VERSION-only** (`PyTorch 2.7 / CUDA 12.8`) | **6/6** | 2/6 |
| MASTER (all rules) | 2/6 | 0/6 |

**Findings:**
- **The version pin is the winner for compilability** (5/6 → 6/6): it steers the model off
  *removed/deprecated* APIs (`.type()`, `.data<>()`, legacy THC, unsync'd warp intrinsics) onto the
  current ATen surface. Cheap, ~1 line.
- **The long "master" rules prompt HURT the 3B model** (compile 2/6, correct 0/6). Over-prescription
  makes a small model attempt complex constructs it can't get right → fewer compiles. Lesson:
  **for small models, anchor the API era (version) and keep the prompt short.**
- Version pin did **not** lift *correctness* (2/6 both) — correctness bugs (wrong math, `input.size()`
  with no dim, inverted bounds) are logic errors no prompt-era fixes; that's the **RFT** target.

## RFT arms (both from SFT-extended, both pushed)
| Arm | Method | Signal | HF |
|---|---|---|---|
| **DPO** | preference (correct+fast ≻ incorrect/slow) per task | pref_acc 0.57, margin +0.28 | `EvanOLeary/laguna-xs2-dense-k8-cuda-dpo` |
| **GRPO-bytask** | group-relative advantage per Task_ID (Dr.GRPO, offline) | reward-weighted; outlier speedups (≤190×) caused loss spikes → clip recommended | `EvanOLeary/laguna-xs2-dense-k8-cuda-grpo` |

Both are **offline / off-policy** (samples are Sakana's, not the policy's) → reweight toward the best
traces, cannot exceed the dataset. True on-policy GRPO would require online inference (vLLM rollouts).

## Models
| Stage | HF |
|---|---|
| SFT | `EvanOLeary/laguna-xs2-dense-k8-cuda-sft` |
| SFT-extended (level_1+2+3, loss→0.086) | `EvanOLeary/laguna-xs2-dense-k8-cuda-sft-v2` |
| RFT · DPO | `EvanOLeary/laguna-xs2-dense-k8-cuda-dpo` |
| RFT · GRPO-bytask | `EvanOLeary/laguna-xs2-dense-k8-cuda-grpo` |
