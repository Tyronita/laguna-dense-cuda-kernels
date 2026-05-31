# Reproducing Laguna Dense — end-to-end, with checkpointing

A permanent, step-by-step guide to reproduce **Laguna Dense** (our ~3.0B all-dense student)
from the **Laguna** teacher (`poolside/Laguna-XS.2`, 33.4B/3.0B-active MoE), then specialize
it for CUDA kernels through SFT → **GRPO** → **DPO**. Every stage lists its exact command,
**where checkpoints land, how often, and how to resume**.

- Pipeline overview & data mixtures: [`training/README.md`](../training/README.md) · [`training/MIXTURES.md`](../training/MIXTURES.md)
- Who-wrote-what + paper provenance (with code): [`docs/PROVENANCE.md`](PROVENANCE.md)

---

## 0. Hardware, environment, conventions

```text
$ nvidia-smi
NVIDIA-SMI 595.71.05   Driver 595.71.05   CUDA 13.2
NVIDIA A100 80GB PCIe   81920MiB   sm_80     # validated here
# recipe runs: H100 PCIe 80GB (all-39-layer recon w/ Adafactor); GB300 for headroom
```
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt
export HF_HOME=/data/hf_cache HF_TOKEN=$(cat ~/.cache/huggingface/token)
export PYTHONPATH=src CUDA_HOME=/usr/local/cuda      # densify.* + nvcc for kernel reward
```
**Optimizer note — Adafactor.** Stage 1 trains 0.98B params; Adam's `m+v` state ≈ 7.9 GB
would not fit beside the 66 GB teacher. Adafactor stores only per-row/col second-moment
factors (≈0 extra memory), so the full 39-layer run fits on 80 GB. See `--optimizer adafactor`.

> **Checkpoint gotcha (applies to every stage):** a saved checkpoint dir must contain the
> custom remote-code files — `modeling_laguna_dense.py`, `configuration_laguna_dense.py`,
> `chat_template.jinja` — or `from_pretrained(..., trust_remote_code=True)` fails to reload it.
> `save_pretrained` keeps the weights+config; copy the three remote-code files in if absent.
> **Read `metrics.jsonl`, not stdout** — tqdm overwrites the loss line in piped logs.

---

## 1. Stage 0 — build dense student + DO-ACP warm-start
```bash
python scripts/00_build_dense_placeholder.py \
    --source-model poolside/Laguna-XS.2 --k-routed 8 \
    --target-dir outputs/laguna-dense-k8-copied-shell
```
**Output:** a ~3.0B / 5.99 GB checkpoint — attention/embeddings/norms/shared-expert copied
from the teacher, routed FFN initialized by **DO-ACP** (select 8 experts → concat → 2.5·α
down-proj). *Checkpoint:* one final dir; no intermediate steps. Push as
`EvanOLeary/laguna-xs2-dense-k8-copied-shell`.

## 2. Stage 1 — reconstruction (the distillation core)
```bash
python scripts/01_train_dense_reconstruction.py \
    --teacher-model poolside/Laguna-XS.2 \
    --student-model outputs/laguna-dense-k8-copied-shell \
    --datasets "GPUMODE/KernelBook:0.40,nvidia/OpenCodeInstruct:0.30,SakanaAI/AI-CUDA-Engineer-Archive:0.20:level_1,ppbhatt500/kernelbook-triton-multiturn-reasoning-traces:0.10" \
    --optimizer adafactor --learning-rate 2e-4 --seq-len 2048 \
    --grad-accum-steps 2 --cosine-weight 0.05 --normalize-loss \
    --max-steps 2000 --save-every 500 --log-every 10 \
    --output-dir outputs/recon_v2
```
**Checkpointing:** writes `outputs/recon_v2/checkpoint-step-{500,1000,1500,2000}` + a final
`checkpoint-final`, `config.json` at start, and per-layer metrics to `metrics.jsonl` every 10
steps. **Resume** by pointing `--student-model` at the latest `checkpoint-step-N`.
**Expected:** V2 loss **0.672 → 0.163** (V1 OpenCode-only: 0.691 → 0.332). Output →
`EvanOLeary/laguna-xs2-dense-k8-recon` (`…-kernelmix` for the V2 mix).

## 3. Stage 3 — SFT (two mixes)
**Mix A — general recovery** (`scripts/02_train_dense_sft.py`, OpenCodeInstruct, seq 8192, lr
5e-5, 500 steps, `--train-norms --train-lm-head`, optional `--kd-*` logit-KD). Supports
`--resume-from-checkpoint` and periodic eval (`--eval-every`).
**Mix B — CUDA** (`scripts/sft_kernel.py`):
```bash
python scripts/sft_kernel.py \
    --student-model EvanOLeary/laguna-xs2-dense-k8-kernelmix \
    --dataset SakanaAI/AI-CUDA-Engineer-Archive --splits level_1,level_2 \
    --max-steps 400 --seq-len 2048 --grad-accum-steps 8 --learning-rate 1e-5 \
    --save-every 200 --output-dir outputs/sft_cuda
```
**Checkpointing:** `checkpoint-step-{200,400}` + `checkpoint-final`; `metrics.jsonl` every 10
steps. Trainable = `routed_dense + lm_head + norms`. This `checkpoint-final` is the **GRPO/DPO
starting point _and_ the frozen reference** for both.

---

## 4. Deep guide — GRPO (`scripts/grpo_kernel.py`)

**What it is.** Online RL with a *verifiable* reward (RLVR): for each task, sample **G** kernels,
score each by actually compiling + running it, and push the policy toward the above-average
samples — **Dr.GRPO** (no std/length normalization) + **DAPO** dynamic sampling (skip groups
with no reward spread), with a **KL anchor** to the frozen SFT model.

**The loop, precisely:**
1. Sample `G` kernels per task at temperature 0.9: `policy.generate(..., num_return_sequences=G)`.
2. Reward each via `kernel_reward.reward_for_text` → `parse .10 / compile .20 / correct .40 /
   speedup .30·min(spd,3)/3` ∈ ~[-0.2, 1.0].
3. **DAPO skip:** if `rewards.std() < 1e-6` the group has no signal → log and skip.
4. **Dr.GRPO advantage:** `adv = r − mean(r)` (no division by std or length).
5. Per sample: `loss = −(adv · logπ / n_tok) + β·KL(π‖π_ref)`, `β=0.02`; backprop; clip 1.0; step.

```bash
python scripts/grpo_kernel.py \
    --model outputs/sft_cuda/checkpoint-final \
    --group-size 6 --max-new-tokens 400 \
    --lr 1e-6 --kl-beta 0.02 --temperature 0.9 --steps 30 \
    --output-dir outputs/grpo
```
**Hyperparameters & why:** `lr 1e-6` (RL is touchy — keep it small); `group-size 6` (more
samples = lower-variance advantage, more compute); `kl-beta 0.02` (anchor to SFT so it doesn't
drift/reward-hack); `temperature 0.9` + `top_k 20` (exploration — too low collapses the group,
too high wastes the budget). Trainable = `routed_dense + lm_head`; the **same SFT model is the
frozen KL reference**.

**Checkpointing & monitoring:** saves `outputs/grpo/checkpoint-step-{10,20,30}` +
`checkpoint-final`. Watch `metrics.jsonl`:
```json
{"step":7,"task":"ReLU","loss":-0.13,"mean_reward":0.42,"max_reward":0.93,
 "compiled":5,"correct":3,"best_speedup":0.97,"elapsed":188.4}
```
- `correct`/`compiled` (out of G) should trend up; `mean_reward` rising = learning.
- Frequent `"skipped":"no-variance"` early = SFT floor too low for that task (all-0 groups);
  expected at the start, should diminish.
**Resume:** restart `--model` from the latest `checkpoint-step-N`.
**Gotchas:** needs `nvcc` (it compiles every sample); `pad_token_id=9` is hard-set for
generation; each step compiles `G` kernels under a 60 s SIGALRM timeout — wall-clock is
dominated by compilation, not the GPU.

---

## 5. Deep guide — DPO (`scripts/dpo_sakana.py`)

**What it is.** *Offline* preference learning — no live compilation. It mines the Sakana
archive's evolutionary trajectory: per task, **prefer the correct + fastest kernel over an
incorrect/slower one**, using the recorded verified labels (`Correct`, `CUDA_Speedup_Native`).

**Pair mining:**
- Group rows by `Task_ID`; `chosen` = correct kernel with the **highest `CUDA_Speedup_Native`**;
  `rejected` = an incorrect kernel (or, if none, a *much-slower* correct one, `<0.5×` chosen).
- Emit up to `pairs-per-task` (prompt, chosen, rejected) triples.

**Loss (Rafailov et al.):**
```
Δ = β[(logπ(chosen) − logπ_ref(chosen)) − (logπ(rejected) − logπ_ref(rejected))]
L = −log σ(Δ)
```
Reference = the frozen SFT model (implicit KL anchor); trainable = `routed_dense + lm_head`.

```bash
python scripts/dpo_sakana.py \
    --model outputs/sft_cuda/checkpoint-final \
    --splits level_1,level_2 --max-tasks 200 --pairs-per-task 8 \
    --beta 0.1 --lr 5e-7 --steps 300 --max-len 1536 \
    --output-dir outputs/dpo
```
**Hyperparameters & why:** `beta 0.1` (preference sharpness vs staying near the reference);
`lr 5e-7` (even gentler than GRPO — DPO over-fits fast); `max-len 1536` (kernels are long; this
truncates both completions consistently).
**Checkpointing & monitoring:** saves **`checkpoint-final` only** (300 steps is short); watch
`metrics.jsonl`:
```json
{"step":10,"loss":0.58,"margin":0.21,"pref_acc":0.7,"elapsed":33.1}
```
- `pref_acc` = fraction of pairs where `Δ>0` (model already prefers chosen). Rising toward
  ~0.7–0.9 = healthy; flat ~0.5 = pairs too noisy / lr too low.
- `margin` (mean `Δ`) growing positive = preference being learned.
**Resume:** re-run from `--model checkpoint-final` (no mid-run checkpoints; lower `--steps` to
checkpoint more often if needed).

---

## 6. Stage → artifact map (checkpoints & HF repos)

| Stage | Local checkpoints | HF repo |
|---|---|---|
| 0 Build | `…/checkpoint` (final only) | `EvanOLeary/laguna-xs2-dense-k8-copied-shell` |
| 1 Reconstruction | `recon_v2/checkpoint-step-{500…2000}` + `-final` | `…-dense-k8-recon` / `…-kernelmix` |
| 3a SFT-A | `sft_opencode/checkpoint-*` | — |
| 3b SFT-B (CUDA) | `sft_cuda/checkpoint-step-{200,400}` + `-final` | `…-dense-k8-cuda-sft` |
| 4 GRPO | `grpo/checkpoint-step-{10,20,30}` + `-final` | `…-dense-k8-cuda-grpo` |
| 5 DPO | `dpo/checkpoint-final` | `…-dense-k8-cuda-dpo` |

## 7. Reproduction checklist
- [ ] Stage 1 reaches V2 loss ≈ 0.16 (kernel mix) and `metrics.jsonl` per-layer MSE drops.
- [ ] SFT-B `checkpoint-final` emits compilable ``` ```cpp ``` blocks (smoke: `scripts/eval_10ops_isolated.py`).
- [ ] GRPO `mean_reward` / `correct` trend up over 30 steps; checkpoints reload (`trust_remote_code`).
- [ ] DPO `pref_acc` rises above ~0.65.
- [ ] Score with `scripts/kernelbench_lite_eval.py` (fast_1) + perplexity — **not** reconstruction MSE alone.
