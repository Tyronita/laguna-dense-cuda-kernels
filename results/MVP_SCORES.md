# MVP eval — 6 tasks, K=4, version-pin prompt, subprocess-isolated

Models: SFT(base) · SFT-ext · GRPO · DPO · Laguna-teacher. Reward = compile + correctness vs PyTorch eager.

## Summary
| Model | compile@4 | correct@4 | mean speedup |
|---|---|---|---|
| SFT | 5/6 | 1/6 | 0.929 |
| SFT-ext | 0/6 | 0/6 | None |
| GRPO | 6/6 | 2/6 | 0.887 |
| DPO | 6/6 | 3/6 | 0.917 |
| Laguna-teacher | 0/6 | 0/6 | None |

> **DPO is the best of our models (6/6 compile, 3/6 correct).** Both GRPO arms (GRPO, DPO) recovered
> the SFT-extended regression (0/6) and beat base SFT. **Teacher 0/6 is a prompt-format artifact** —
> the short version-pin prompt fits our chat-tuned models; Laguna scored 4/6 in the earlier head-to-head
> with its own format. Not a clean DPO>teacher claim.

## Per-task (correct on ≥1 of K=4)
| task | SFT | SFT-ext | GRPO | DPO |
|---|---|---|---|---|
| relu | ✅ | — | ✅ | ✅ |
| tanh | — | — | ✅ | — |
| sigmoid | — | — | — | ✅ |
| gelu | — | — | — | — |
| abs | — | — | — | — |
| silu | — | — | — | ✅ |

Full per-sample code + compile errors: `mvp_6task_full.md`.
