# Consolidation / PR plan — landing all branches to `main` (+ CUDA separation)

## Repos & roles (the separation)
- **`cm2435/laguna-xs2-expert-coactivation-scheduling`** — the **research monorepo**: MoE→dense
  **densification core** + **agentic SWE-bench** (rollouts, pool backend, swebench tasks) + CUDA
  experiments. Upstream where most code originates.
- **`Tyronita/laguna-dense-cuda-kernels`** — the **CUDA productization**: densify core + the
  `000–004` CUDA post-training pipeline + eval, packaged clean (no agentic cruft).

**The repo boundary *is* the primary CUDA separation.** cm2435 keeps the agentic + research breadth;
Tyronita is the CUDA-kernel slice.

## Branch / PR inventory
### cm2435 (`main` = `5667f02`, far behind — a May-29 spike merge)
| PR / branch | → base | size | status | contents |
|---|---|---|---|---|
| **#6** `recipe/paper-aligned-densification-kernel-data` | `main` | +46.9k / 1198 files | ⚠️ **conflicts** | densification recipe, DO-ACP, expert analysis, CUDA tooling — the spine |
| **#5** `docs/dense-placeholder-training-plan` | `main` | +53.6k / 1250 files | mergeable | dense recovery pipeline, narrative, swebench rollout tasks (agentic) |
| **#7** `docs/training-provenance-and-cleanup` | `recipe` (#6) | +0.7k / 19 files | mergeable | provenance/investigation docs + `000–004` renames (stacked on #6) |
| `jessica/sft-recovery` | — | — | no PR | SFT-recovery writeup + HumanEval |
| `feat/kernel-sft-rft-tooling` | — | — | no PR | likely subsumed by #6 |
| `spike/*` | — | — | superseded | early harness / permission spikes |
### Tyronita (`main`, current)
| PR / branch | → base | status | contents |
|---|---|---|---|
| **#5** `kernelbench-l1-eval` | `main` | mergeable | L1 eval results (large) |
| `consolidation/readme-and-pr-plan` | `main` | **new PR** | grpo-convergence (isolated reward) + README layout + this plan + double-prefix bugfix |
| *(merged)* `feat/end-to-end-training-recipe`, `docs/reproduce-provenance`, `sync/numbered-training-pipeline` | — | merged | scripts + docs already on `main` |

## Recommended merge order
**cm2435** (the hard one — `main` is far behind and two huge PRs both target it):
1. **Decide the data policy first.** #5 and #6 each add ~1,200 files (rollouts / swebench tasks / figures).
   Either (a) accept the monorepo size, or (b) split each into a **code+docs PR** (small) and put the
   rollout/task data behind `.gitignore`/LFS. **Recommended: (b)** for a clean `main`.
2. **Land #6 (recipe) first** — the densification spine. **Rebase on `main`** to clear the conflict
   (likely README/docs both touched).
3. **Retarget #7 → `main`** once #6 lands (it is stacked on #6).
4. **#5 (dense-placeholder / agentic recovery)** — the agentic half; can land independently.
5. **`jessica/sft-recovery`** → fold its writeup into docs or open a small PR; **close the spikes**.
**Tyronita**:
1. Land the **consolidation PR** (grpo convergence + README + this plan + bugfix).
2. **#5 `kernelbench-l1-eval`** — land the results, or keep as a results branch (large; consider docs-only).

## CUDA-focused separation (the ask), inside each repo
| Group | Paths |
|---|---|
| **Densify core** (stages 0–1) | `scripts/000_build_dense_placeholder.py`, `scripts/001_train_dense_reconstruction.py`, `src/densify/{densify_layer,reconstruction,dense_checkpoint/*}.py` |
| **CUDA post-training** (stages 2–4) ⭐ | `scripts/002_sft_*.py`, `scripts/003_grpo.py`, `scripts/003_grpo_offline.py`, `scripts/004_dpo.py`, `src/densify/kernel_reward.py` |
| **CUDA eval & ablations** | `scripts/eval_worker.py`, `kernelbench_lite_eval.py`, `eval_10ops_isolated.py`, `head_to_head.py`, `ablate_*.py` |
| **Agentic** (cm2435 only) | `src/densify/{swebench,coding_harness,rollout_sft,pool_backend,…}` |

**Optional follow-up "restructure" PR:** move the CUDA post-training + eval into
`scripts/{densify,posttrain,eval}/` subdirs (the `000–004` number stays as the in-group prefix).

## Open items
- `reconstruction_data.py` is **intentionally diverged** (Tyronita agentic tool-call formatting vs
  cm2435's `_coerce_messages`, imported by ~18 consumers) — unify only if needed.
- **`000X_` double-prefix doc typo** (from an over-eager rename `sed`) — fixed in the Tyronita
  consolidation PR; **same fix applied to cm2435 docs here**.
- **Repo size** from rollout/task data — decide `.gitignore`/LFS before landing #5/#6.
