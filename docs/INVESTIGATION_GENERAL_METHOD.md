# Investigation — the general MoE→dense method: commit lineage, training patterns, random-init vs lift-and-shift (confirmed)

*Group research notes. Teacher = `poolside/Laguna-XS.2` (33.4B/3.0B-active MoE); student =
Laguna Dense (~3.0B all-dense). This documents what the method is, how it evolved across
commits, the two competing init strategies, and an **empirical confirmation**.*

## 1. The general method
1. **Copy the shell** of the teacher (attention, embeddings, norms, shared expert) verbatim.
2. **Replace each routed MoE block** (256 experts, top-8) with **one dense SwiGLU** (`routed_dense`).
3. **Initialize** that dense FFN — either **random** or **lift-and-shift (DO-ACP)**.
4. **Reconstruct**: teacher-forced, per-layer `MSE + 0.05·cos` (÷ mean y²), train **`routed_dense` only**.
5. **Specialize**: SFT → GRPO → DPO for CUDA kernels.

## 2. Commit lineage (in order)
| Commit | UTC | Author | What changed (training pattern) |
|---|---|---|---|
| `e5a4bba` | 05-29 17:43 | Charlie (cm) | RFC: choose **prune-and-distill** (DO-ACP) over MergeMoE/MoE-Pruner/FP8 |
| `831bd3e` | 05-29 20:52 | Charlie (cm) | Dense placeholder; **`--init random`** is the default (random `routed_dense`) |
| `b8e2aaf` | 05-29 21:25 | Charlie (cm) | Reconstruction trainer (teacher-forced MSE) — trains **981.5 M** `routed_dense` |
| **`8a687e2`** | 05-29 22:54 | Evan/Tyronita | **DO-ACP warm-start (`--init selected-concat`) = the lift-and-shift** + per-layer loss norm |
| `3ff0c5c`→`b8e50c6` | 05-29 22:28→05-30 01:59 | Evan/Tyronita | Paper-aligned recipe, **Adafactor**, kernel-data mixture |
| `f8fc9ce…` | 05-30 03:43 | Evan/Tyronita | expert-activation analysis (~158 eff/layer, Gini 0.528, coactiv 2.09×) |
| `b66eec5`,`ce390cd` | 05-30 05:42–06:51 | Evan/Tyronita | kernel SFT/GRPO/DPO/eval |

**Reading:** Charlie built the *framework* (RFC + placeholder + reconstruction) with **random**
init; Evan added the **lift-and-shift (DO-ACP) init** and the kernel-anchored recipe on top.

## 3. The two training patterns — random init vs lift-and-shift
| | Random init (Charlie, `main` default) | Lift-and-shift / DO-ACP (Evan) |
|---|---|---|
| `routed_dense` start | `normal(0, 0.02)` — from scratch | **8 real experts selected (DO-ACP) and concatenated** (gate/up rows, down cols × 2.5·α) |
| Trainable params | 981.5 M (32.8%) | 981.5 M (32.8%) — **identical count** |
| Starting function | far from the teacher's MoE block | **near** the teacher's MoE block |
| Flag | `--init random` | `--init selected-concat` |

**Key point:** the two differ **only in initialization**, not in *what* trains or *how many*
params. Both then run the same teacher-forced reconstruction.

## 4. Empirical confirmation
Using the **real `densify_layer.py`** (DO-ACP code) on a synthetic 64-expert top-8 MoE block,
measuring how well a dense FFN reproduces the MoE block output **with no training** (i.e., the
quality of the *starting point*):

| Init (no training) | rel-L2 ↓ | cosine ↑ |
|---|---|---|
| random (Charlie's default) | 0.853 | 0.542 |
| lift-shift: frequency | 0.839 | 0.565 |
| lift-shift: ACP | 0.818 | 0.587 |
| **lift-shift: DO-ACP** | **0.818** | **0.587** |

**Confirmed:** lift-and-shift gives a **measurably better starting point than random** (lower
error, higher cosine), and the ordering **DO-ACP ≥ ACP ≥ frequency ≥ random** matches the KRAFTON
paper (which reports *scoring dominates*, ~5.7 pp).

**Honest caveats:**
- This is a **synthetic** MoE (random experts), **not** the gated 33B teacher. The real Laguna
  has far more expert redundancy (Gini 0.528, coactivation 2.09×), so **DO-ACP's diversity edge
  over plain ACP/frequency should be larger on the real model** (here ACP≈DO-ACP because random
  experts aren't strongly grouped).
- Absolute rel-L2 stays high (~0.82) because a dense FFN applies **all 8 experts to every token**
  (no routing), so it cannot match the per-token top-8 mixture *without training*. The claim being
  confirmed is **"better starting point for reconstruction,"** not zero-shot equality.

## 5. Verdict — is the training method confirmed?
✅ **Yes, the method is sound and correctly implemented:**
- Lift-and-shift (DO-ACP warm-start) **beats random init** at initialization (this test).
- The selection ordering **DO-ACP ≥ ACP ≥ frequency ≥ random** holds (paper-consistent).
- The objective (teacher-forced MSE+cos, frozen shell, Adafactor) is consistent with
  KRAFTON (densification) + FitNets (feature hints) + RADLADS staging.

⚠️ **Open item (recommended next):** we do **not** have a measured A/B of *random vs lift-and-shift
reconstruction-loss curves on the real teacher* in the logs — the docs' "single biggest convergence
lever" is asserted + paper-backed + supported by this synthetic test, but a real-teacher A/B (two
short reconstruction runs, `--init random` vs `--init selected-concat`, same data/steps) would
close it definitively. The code path exists (`000_build_dense_placeholder.py --init {random,selected-concat}`).
