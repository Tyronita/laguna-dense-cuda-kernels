# Laguna Dense — training plan, provenance & trainable-parameter ledger

**Laguna** = teacher `poolside/Laguna-XS.2` (33.4B total / 3.0B active MoE, 256 experts, top-8).
**Laguna Dense** = our student: **2,996,678,656 params (~3.0B), all dense**. This is the single
authoritative reference for *what we did, who did it, how many parameters trained at each stage,
and which paper each step comes from*.

---

## 1. The model (real `print(model)` numbers)

| Module | Params | % | Stage-1 recon |
|---|---|---|---|
| embed_tokens | 205.5 M | 6.9% | frozen |
| 40 × self_attn (GQA 48–64 / 8, head_dim 128) | 1431.0 M | 47.8% | frozen |
| layer-0 **dense** mlp (width 8192) | 50.3 M | 1.7% | frozen |
| **routed_dense** (39 densified layers) | **981.5 M** | **32.8%** | **TRAIN** |
| shared_experts (39 layers) | 122.7 M | 4.1% | frozen |
| all RMSNorms | 0.2 M | 0.0% | frozen |
| lm_head | 205.5 M | 6.9% | frozen |
| **TOTAL** | **2.997 B** | | |

> Only layer 0 is a plain dense MLP; layers 1–39 are the densified MoE blocks (`routed_dense` +
> kept `shared_experts`). Attention head count varies per layer (48 or 64 query heads, 8 KV).

---

## 2. Trainable parameters at **each stage** (the ledger you asked for)

| Stage | Script | Trainable modules | Trainable params | % of 3.0B |
|---|---|---|---|---|
| **0 · Build + warm-start** | `000_build_dense_placeholder.py` | — (init only, 0 steps) | 0 | 0% |
| **1 · Reconstruction (distillation)** | `001_train_dense_reconstruction.py` | `routed_dense` | **981.5 M** | **32.8%** |
| **2 · SFT-A (general)** | `002_sft_general.py` (`--train-norms --train-lm-head`) | `routed_dense + RMSNorms + lm_head` | **1187.2 M** | 39.6% |
| **2 · SFT-B (CUDA)** | `002_sft_cuda.py` | `routed_dense + lm_head + RMSNorms` | **1187.2 M** | 39.6% |
| **3 · GRPO** | `003_grpo.py` | `routed_dense + lm_head` | **1187.0 M** | 39.6% |
| **4 · DPO** | `004_dpo.py` | `routed_dense + lm_head` | **1187.0 M** | 39.6% |

**Key fact:** every stage trains **only the FFN (`routed_dense`)** plus, after reconstruction,
the `lm_head` (and norms). Attention, embeddings, and the shared expert are **always frozen** —
they are copied verbatim from the teacher and never updated.

### Did Charlie's original `main` approach do this? How many trainable params?
- **Charlie's base reconstruction** (`b8e2aaf`, `freeze_for_dense_reconstruction`) trained the
  **same 981.5 M `routed_dense`** params (32.8%). The trainable *count is identical*.
- **The difference is the initialization, not the count.** Charlie's build driver defaults to
  **`--init random`** (`copied_shell_random_routed`): the dense FFN starts from **random weights**.
  **DO-ACP "lift-and-shift" (`--init selected-concat`) was added by Evan/Tyronita** (`8a687e2`) —
  it changes *where training starts*, not *how many params train*. So: **Charlie's `main` did NOT
  do DO-ACP**; it trained 981.5 M randomly-initialized FFN params. Evan's DO-ACP warm-start trains
  the same 981.5 M but from a much better starting point.

---

## 3. What DO-ACP and "warm-start" actually are (+ step counts)

**Two different "warm" things — don't conflate them:**

- **DO-ACP warm-start = *weight initialization* (0 training steps).** Before any training, fill the
  dense FFN with a good guess instead of random noise: per layer, **score** the 256 experts (ACP),
  **select** the best 8 that are also diverse (D-optimal), and **concatenate** their weight matrices
  into one SwiGLU. This is an *init*, so it costs **0 steps** — it just sets the starting weights.
  *(`densify_layer.select_do_acp` + `build_dense_ffn`; from the KRAFTON paper.)*
- **LR warmup = ramping the learning rate over the first N steps.** The 3B recipe scripts **do not
  use LR warmup** — reconstruction uses Adafactor at a constant `lr 2e-4`; SFT/GRPO/DPO use constant
  AdamW. (LR warmup appears only in the separate 5B branch.)

**Step counts per stage (as run):**

| Stage | Steps | Tokens / data |
|---|---|---|
| 0 Build (DO-ACP warm-start) | **0** (init) | — |
| 1 Reconstruction | **~2000** (V2; a headline run ≈1520) | ~6.2 M teacher-forced |
| 2 SFT-B (CUDA) | **400** | Sakana, seq 2048 |
| 2 SFT-A (general) | **500** | OpenCodeInstruct, seq 8192 |
| 3 GRPO | **30** (group-size 6) | live KernelBench reward |
| 4 DPO | **300** | Sakana preference pairs |

---

## 4. What K=8, the shared expert, and the "lift-and-shift" are

- **K = 8.** The teacher routes each token to **8 of 256** experts (top-8). When we collapse the
  MoE into one dense FFN, **K is how many experts we keep/merge.** We set **K = 8** so the dense FFN
  has the same per-token capacity the router used. Dense width = **K × moe_intermediate(512) = 4096**.
  C4 diagnostics show ~158 *effective* experts/layer, so **K is a planned sweep (8 → 16 → 32)** —
  K=8 matches top-8 but under-fits the deep layers.
- **Shared expert.** Laguna's MoE has two FFN types per layer: the **256 routed experts** (only
  top-8 fire, they specialize) **and one always-on "shared expert"** (every token passes through it;
  it captures the common computation). It is a plain SwiGLU of width 512. In densification it is
  **already dense**, so we **keep it verbatim** (copied, frozen) — only the routed part is collapsed.
- **How the experts were "lifted and shifted."** DO-ACP picks 8 experts per layer; then
  `build_dense_ffn` literally **stacks their weight matrices** into one FFN: concat the 8 `gate`
  rows → `[4096, 2048]`, concat the 8 `up` rows → `[4096, 2048]`, and concat the 8 `down` columns —
  each scaled by **2.5·α_e** (the router's ×2.5 scaling × that expert's marginal routing weight) —
  → `[2048, 4096]`. So real expert weights are *lifted out of the MoE and shifted into the dense
  FFN as its starting point*. **Charlie's `main` skipped this** (random init); the lift-and-shift is
  Evan's DO-ACP warm-start.

---

## 5. Did we do logit-KD ("KD-Logits")? — No (honest correction)

**Logit-KD** = train the student so its **output logits** match the teacher's full probability
distribution, via `KL(student ‖ teacher)`. It operates on the *final vocab logits*.

**What we actually did is *activation* / *feature* reconstruction, not logit-KD.** Stage 1 matches
the teacher's **per-layer MLP-block outputs** with `MSE + 0.05·cos` (FitNets-style feature hints) —
*not* the output logits. **Logit-KD was documented as a planned Stage 2 but was *not run*:**
`002_sft_*.py` exposes an optional `--kd-weight` that **defaults to 0** (off), and no logit-KD run
exists. So the honest pipeline is: **activation-reconstruction → SFT → GRPO → DPO**. Logit-KD
remains a future lever (it's the one RADLADS calls "step 2" and that Hinton-KD describes).

---

## 6. Commit ledger (verified) + results

| # | Piece | Author | Commit | UTC | Result |
|---|---|---|---|---|---|
| RFC | MoE→dense method choice (prune-and-distill) | Charlie (cm) | `e5a4bba` | 05-29 17:43 | chose DO-ACP over MergeMoE/MoE-Pruner/FP8 |
| 0 | Dense placeholder (`build_placeholder.py`, `config.py`; `shared_expert="kept"`, `--init random`) | Charlie (cm) | `831bd3e` | 05-29 20:52 | `…-dense-k8-copied-shell`, 3.0B / 5.99 GB |
| 1 | Base reconstruction (`reconstruction.py`, trainer; teacher-forced MSE) | Charlie (cm) | `b8e2aaf` | 05-29 21:25 | V1 loss 0.691→0.332; **trains 981.5M** |
| 0+1 | **DO-ACP warm-start** + per-layer loss norm (÷mean y²) | Evan/Tyronita | `8a687e2` | 05-29 22:54 | lift-and-shift init = "biggest convergence lever" |
| 1 | Paper-aligned recipe + **Adafactor** + kernel mix | Evan/Tyronita | `3ff0c5c`→`b8e50c6` | 05-29 22:28→05-30 01:59 | fits 39 layers on 80 GB; V2 loss 0.672→**0.163** |
| analysis | expert-activation / co-activation | Evan/Tyronita | `f8fc9ce…` | 05-30 03:43 | ~158 eff-experts/layer, Gini 0.528, coactiv 2.09× |
| 2–4 | kernel SFT / GRPO / DPO / eval + `MODEL_CHANGES.md` | Evan/Tyronita | `79175e8`,`b66eec5`,`ce390cd` | 05-30 02:08–06:51 | HumanEval 1/10; ReLU 3/4; 11× smaller, +26% faster |

---

## 7. Citations — training plan & model architecture

**Model architecture (inherited from the teacher — not our work):**
- Shared expert → **DeepSeekMoE**, Dai et al. 2024 — [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)
- SwiGLU FFN → **GLU Variants**, Shazeer 2020 — [arXiv:2002.05202](https://arxiv.org/abs/2002.05202); SiLU — [arXiv:1702.03118](https://arxiv.org/abs/1702.03118)

**Training plan (what we implemented):**
- **DO-ACP densification (core)** → "Pruning & Distilling MoE into Dense LMs" (Kim et al., 27 May 2026) — [arXiv:2605.28207](https://arxiv.org/abs/2605.28207)
- **Reconstruction objective** → **FitNets** feature hints, Romero et al. 2014 — [arXiv:1412.6550](https://arxiv.org/abs/1412.6550); *staging only* from **RADLADS** — [arXiv:2505.03005](https://arxiv.org/abs/2505.03005) ⚠️ (RADLADS swaps attention→linear, which we do **not** do — we keep GQA; we reuse only its align→KD→SFT staging)
- **Optimizer** → **Adafactor**, Shazeer & Stern 2018 — [arXiv:1804.04235](https://arxiv.org/abs/1804.04235)
- **Post-training** → **GRPO** [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) · **Dr.GRPO** [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) · **DAPO** (dynamic sampling) [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) · **RLVR/DeepSeek-R1** [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) · **DPO** [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) · **KernelBench** [arXiv:2502.10517](https://arxiv.org/abs/2502.10517)
- **Logit-KD (planned, not run)** → Hinton et al. 2015 — [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- Dataset = **Sakana AI CUDA Engineer Archive** (`SakanaAI/AI-CUDA-Engineer-Archive`, 2025) — ⚠️ arXiv ID unconfirmed (the earlier `2502.14333` was wrong — that's a math-reasoning survey).

---

## 8. Distributed training data & data-cleaning

Reconstruction depends **only on what text drives the frozen teacher's forward pass**, so the data
mix is a direct quality lever (it sets which experts activate → which must be reconstructed).

**Reconstruction mixes** (`001_train_dense_reconstruction.py --datasets "name:weight[:split]"`,
seeded streaming interleave):
- **V1** — `nvidia/OpenCodeInstruct` only → loss 0.691→0.332.
- **V2** — `GPUMODE/KernelBook:0.40, nvidia/OpenCodeInstruct:0.30, SakanaAI/AI-CUDA-Engineer-Archive:0.20:level_1, …triton-traces:0.10` → loss 0.672→**0.163** (~2× lower).

**SFT mixes:** Mix A = `nvidia/OpenCodeInstruct` (general recovery); Mix B = `SakanaAI/AI-CUDA-Engineer-Archive` (`level_1,level_2`, `Correct==True`, `PyTorch_Code_Module→CUDA_Code`).
**DPO pairs:** Sakana traces, correct+fastest (`CUDA_Speedup_Native`) ≻ incorrect/slower.

**Data-hygiene items to clean on cm2435:**
1. **Sakana split** must be `level_1/level_2`, **not `train`** — the wrong-split bug silently drops all CUDA rows (already fixed in `reconstruction_data` + the `:split` syntax; ensure every config uses it).
2. **Field handler** (`reconstruction_data.format_sft_row`) must recognize the `query`/`kernel`/`code` schemas (KernelBook/CUDA) or those rows are skipped → mix silently degrades to OpenCode-only.
3. **`Correct==True` filter** on Sakana for SFT/DPO (hold out `CUDA_Speedup_Native`/`NCU_Profile`/`Clang_Tidy` for the GRPO reward, not for SFT).
4. **De-dup & decontaminate** against KernelBench tasks before GRPO to avoid reward-hacking on seen solutions.

---

## 9. Is this trainable scope typical for SFT?

**No — it's a deliberate *partial* fine-tune, not standard SFT.** Three regimes for comparison:

| Regime | What trains | Typical % | Used where |
|---|---|---|---|
| **Full SFT** | *all* parameters | 100% | the default for instruction-tuning (e.g. most code models) |
| **PEFT / LoRA** | a small low-rank add-on, base frozen | <1% | cheap adaptation |
| **Ours** | `routed_dense` (+`lm_head`,+norms); attention/embed/shared **frozen** | **~33–40%** | dictated by the densification constraint |

**Why ours is unusual *and* intentional:** in densification the **attention, embeddings, and shared
expert were copied verbatim from the teacher and are already "correct."** Freezing them (a) preserves
the teacher's behavior and avoids catastrophic forgetting, (b) keeps teacher+student+optimizer inside
80 GB, and (c) focuses the gradient on the *only* part we changed — the reconstructed FFN. So it's a
**selective full-rank fine-tune of the modified subnetwork**: heavier than LoRA, lighter than full SFT.
Standard SFT would unfreeze everything; we deliberately do not.

## 10. What is "warmup" (and how is it different from "warm-start")?

**Learning-rate warmup** = for the first **N steps**, ramp the learning rate **from ~0 up to the target**
(usually linearly), then hold or decay it.
- **Why:** at the start, weights are fresh and an adaptive optimizer's moment estimates (Adam's `m`/`v`)
  are unreliable; jumping straight to full LR can cause a **loss spike / divergence** (large early
  `grad_norm`). Warmup lets the optimizer statistics settle and the model ease in.
- **Typical values:** `warmup_ratio` ≈ 1–5% of total steps, or a fixed `warmup_steps` (≈ 8–100).
- **In *our* recipe:** the 3B scripts use **constant LR, no warmup** (Adafactor `lr 2e-4`; AdamW constant).
  The separate 5B branch *did* use `warmup_steps 8` — after a too-short 2-step warmup spiked `grad_norm`
  to 4.56.
- **Not the same as "warm-start":** **warmup** ramps the **learning rate over steps**; **DO-ACP
  warm-start** sets the **initial weights** (0 steps). Both contain "warm," but one is an optimizer
  schedule and the other is an initialization.
