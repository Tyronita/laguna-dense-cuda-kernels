# Provenance — who introduced what, with code & papers

**Laguna** = the Poolside `Laguna-XS.2` 33.4B MoE we distil **from**. **Laguna Dense** = our
~3.0B all-dense student. Authorship below is **verified from the introducing commit** of each
file; paper IDs were **opened and verified** (✓). Repo: branch
`recipe/paper-aligned-densification-kernel-data` @ `7b9dd15`.

## Where DO-ACP comes from
DO-ACP is **the KRAFTON paper's method** — *Pruning and Distilling MoE into Dense LMs* (Kim et
al., arXiv:2605.28207), its named best expert-selection scoring. **Charlie (cm)** chose the
prune-and-distill approach (RFC `e5a4bba`); **Evan/Tyronita** implemented DO-ACP for Laguna's
sigmoid-router MoE (`densify_layer.py`, `8a687e2`). Algorithm = paper; implementation = us.

---

## The table (author · commit · timestamp · code · paper)

### 1. Shared expert — Poolside (teacher) · — · inherited
Always-on dense MLP added to the routed mixture; we **keep it verbatim** and only learn `routed_dense`.
```python
# build_placeholder.py — the dense block (router removed, shared kept)
def forward(self, hidden_states):
    routed_output = self.routed_dense(hidden_states) * self.routed_scaling_factor  # 2.5×
    shared_output = self.shared_experts(hidden_states)                              # KEPT
    return routed_output + shared_output
```
**Paper:** DeepSeekMoE, Dai et al. 2024 — [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) ✓ (shared-expert isolation).

### 2. MoE→dense method-selection RFC — Charlie (cm) · `e5a4bba` · 2026-05-29 17:43 UTC
Surveyed prune-and-distill vs MergeMoE / MoE-Pruner / FP8 → chose **prune-and-distill (DO-ACP)**.
**Result:** set the entire densification direction.
**Paper:** Kim et al. 2026 — [arXiv:2605.28207](https://arxiv.org/abs/2605.28207) ✓.

### 3. Dense placeholder tooling — Charlie (cm) · `831bd3e` · 2026-05-29 20:52 UTC
`build_placeholder.py`, `config.py` — copies the teacher shell, declares the dense conversion.
```python
# config.py — the conversion is declared structurally
self.dense_routed_intermediate_size = k_routed * expert_intermediate_size   # 8×512 = 4096
self.moe_dense_conversion = {"kind": "routed_moe_to_dense_swiglu",
                             "shared_expert": "kept",
                             "placeholder_weights": "copied_shell_random_routed"}
```
**Result:** `laguna-xs2-dense-k8-copied-shell`, ~3.0B / 5.99 GB.
**Papers:** Kim et al. 2026 ✓ (prune-and-concat target) · shared-kept ← DeepSeekMoE 2401.06066 ✓.

### 4. Base reconstruction pipeline — Charlie (cm) · `b8e2aaf` · 2026-05-29 21:25 UTC
`reconstruction.py`, `train_dense_reconstruction.py` — teacher-forced per-layer MSE.
```python
# reconstruction.py — MSE is taken from the teacher's MLP-block output, teacher-forced
pred   = student_mlp(x)                         # x = teacher's MLP input (hooked)
mse    = (pred - target).pow(2).mean(-1)        # target = teacher MLP output  y_l
loss   = mse / (target.pow(2).mean(-1) + 1e-6)  # ÷ energy   (--normalize-loss)
loss   = loss + 0.05 * (1 - cos(pred, target))  # + cosine term
```
**Result:** V1 (OpenCode) loss **0.691 → 0.332**; deep L28–39 0.232 → 0.025.
**Papers:** feature-hint distillation — **FitNets**, Romero et al. 2014 — [arXiv:1412.6550](https://arxiv.org/abs/1412.6550) ✓ · *staging* from **RADLADS** rep-alignment step — [arXiv:2505.03005](https://arxiv.org/abs/2505.03005) ✓ **(staging only — see note)**.

### 5. DO-ACP warm-start + per-layer loss norm — Evan/Tyronita · `8a687e2` · 2026-05-29 22:54 UTC
`densify_layer.py` — score (ACP) → diversity-aware select (D-optimal) → concat.
```python
# ACP = activation-weighted conditional prob:  CP_e · √ E‖f_e‖²
def acp_scores(routing, experts):
    return routing.cp * torch.sqrt(experts.out_norm_sq)

# DO = greedy D-optimal on the importance-weighted expert-output Gram
def select_do_acp(routing, experts, k):
    I = acp_scores(routing, experts).double(); sqrtI = I.clamp(min=1e-12).sqrt()
    K = (sqrtI[:,None] * sqrtI[None,:]) * experts.gram          # K_ij = √(I_iI_j)·G_ij
    Kr = K + (K.diagonal().mean()/k) * torch.eye(len(I))
    selected = []
    for _ in range(k):                                          # maximize log det
        best = max(remaining, key=lambda e: slogdet(Kr[selected+[e]][:, selected+[e]]))
        selected.append(best)
    return selected

# concat selected experts; fold 2.5×α (routed-scaling × marginal weight) into down-proj
down_blocks.append(down[e] * (mlp.routed_scaling_factor * alpha[e]))
```
**Result:** warm-start init = "single biggest convergence lever"; per-layer ÷mean(y²) balances deep vs shallow.
**Paper:** Kim et al. 2026 — [arXiv:2605.28207](https://arxiv.org/abs/2605.28207) ✓ (ACP + D-optimal selection; D-optimality = classical optimal experimental design).

### 6. Paper-aligned recipe + Adafactor + kernel-data mixture — Evan/Tyronita · `3ff0c5c` (22:28) → `b8e50c6` (2026-05-30 01:59 UTC)
```python
# train_dense_reconstruction.py — sublinear-memory optimizer (fits 39 layers on 80GB)
optimizer = Adafactor(trainable_params, lr=2e-4,
                      scale_parameter=False, relative_step=False, warmup_init=False)
# --datasets "name:weight[:split]"  → seeded streaming interleave (mixed_rows)
```
**Result:** Adafactor enables all-39-layer run; **V2 kernel mix loss 0.672 → 0.163** (~2× lower than V1; headline 0.237 → 0.0266, −88.8%).
**Papers:** **Adafactor**, Shazeer & Stern 2018 — [arXiv:1804.04235](https://arxiv.org/abs/1804.04235) ✓ · kernel-mix = **original** (domain coverage). Dataset = Sakana *AI CUDA Engineer Archive* (2025) — ⚠️ **arXiv ID unconfirmed; earlier `2502.14333` was WRONG** (a math-reasoning survey).

### 7. MODEL_CHANGES.md + expert-activation analysis + post-training — Evan/Tyronita · `79175e8` (02:08) / `f8fc9ce` (03:43) / `b66eec5` (05:42) / `ce390cd` (06:51 UTC)
```python
# kernel_reward.py — verifiable shaped reward (RLVR)
rew  = 0.10 if parsed else -0.10
rew += 0.20 if compiled else 0.0
rew += 0.40 if correct  else 0.0
rew += 0.30 * min(speedup, 3.0)/3.0 if correct else 0.0
# grpo_kernel.py — Dr.GRPO advantage + DAPO dynamic sampling
if rt.std() < 1e-6: continue                 # DAPO: skip zero-variance group
adv  = rt - rt.mean()                         # Dr.GRPO: no std/length norm
loss = -(adv[g] * lp / ntok) + kl_beta * kl   # PG + KL anchor to SFT ref
# dpo_sakana.py — DPO (Rafailov) on Sakana traces
delta = beta * ((lc - rc) - (lr - rr)); loss = -F.logsigmoid(delta)
```
**Result:** C4 analysis — **~158 eff-experts/layer, Gini 0.528, coactivation 2.09×**; kernel-SFT HumanEval 1/10; ReLU 3/4·Tanh 3/4·**Sigmoid 0/4**, ~2/5 k=1; **11× smaller, 12× less VRAM, +26% faster** vs teacher.
**Papers:** GRPO/DeepSeekMath [2402.03300](https://arxiv.org/abs/2402.03300) ✓ · Dr.GRPO [2503.20783](https://arxiv.org/abs/2503.20783) ✓ · DAPO [2503.14476](https://arxiv.org/abs/2503.14476) ✓ · RLVR/DeepSeek-R1 [2501.12948](https://arxiv.org/abs/2501.12948) ✓ · KernelBench [2502.10517](https://arxiv.org/abs/2502.10517) ✓ · DPO/Rafailov [2305.18290](https://arxiv.org/abs/2305.18290) ✓ · logit-KD option ← Hinton et al. 2015 [1503.02531](https://arxiv.org/abs/1503.02531).

---

## ⭐ KRAFTON vs RADLADS — the two cited papers, read in full

**KRAFTON — *Pruning and Distilling MoE into Dense LMs*** (Kim, Yun, Kim, Kim, Bae, Cho — arXiv:2605.28207, 27 May 2026). **This is the method Laguna densification actually implements.** It proposes: score experts → select top-K → group → **concatenate into one dense FFN** → refine by KD from the MoE teacher; it finds **scoring dominates grouping (~5.7 pp vs ~1 pp)**, **pure-prune + DO-ACP is best**, down-proj magnitude scaling preserves output magnitude, and MoE→dense beats dense-pruning by **+6.3 pp** after ~4B-token KD (1.6× faster). → drives commits `e5a4bba`, `831bd3e`, `8a687e2`.

**RADLADS — *Rapid Attention Distillation to Linear Attention Decoders at Scale*** (Goldstein, Alcaide, Lu, Cheah — arXiv:2505.03005, May 2025). Its real contribution is converting **softmax → *linear* attention** in **350–700M tokens** via a **3-stage protocol (align reps → logit-KD → SFT)**. ⚠️ **Laguna does NOT do the attention swap** — it keeps GQA verbatim and changes the FFN/MoE. We borrow **only RADLADS's 3-stage staging + tiny-token-budget philosophy**, applied to a different architectural axis. Cite it for *recipe shape*, not method.

---

## Corrections folded in
1. **Sakana arXiv `2502.14333` removed** — verified unrelated (math-reasoning survey).
2. **RADLADS relabeled** — staging only; it's an attention-linearization paper, which we don't use.
3. **`2605.28207` confirmed genuine** (Kim et al. 2026); "KRAFTON" is the repo's nickname, not the title.

*Confidence: ✓ = paper opened and verified this pass. `1503.02531` foundational, not re-fetched. Architecture rows (shared expert, SwiGLU, SiLU) are the teacher's, inherited from Poolside.*
