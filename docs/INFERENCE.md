# Inference optimization — measured (tok/s + TTFT) & serving-backend support

**Model:** `EvanOLeary/laguna-xs2-dense-k8-cuda-grpo` (~3.0 B dense). **HW:** A100 80 GB.
**Measurement:** single sequence (batch 1), 77-token prompt, greedy, 128 new tokens; `decode` =
steady-state tok/s (excludes prefill), `TTFT` = prefill + first token. *(Run under GPU contention
with a separate 67 GB teacher job — the **ratios** are solid; absolute tok/s ≈ ±10 %.)*

## Results

| Backend | TTFT | decode tok/s | vs eager | Notes |
|---|---|---|---|---|
| **HF transformers, bf16 eager** | 71 ms | **15.4** | 1.0× | baseline (`attn_implementation="sdpa"`) |
| **HF transformers + `torch.compile`** | **44 ms** | **32.9** | **2.1×** | `mode="default"` ✅ · **`max-autotune` ❌** (CUDA-graphs vs `generate` cache → `RuntimeError: accessing tensor overwritten by CUDAGraphs`) |
| **vLLM 0.22 (native + transformers backend)** | — | — | — | ❌ **does not load** — see below |
| **SGLang** | — | — | — | not installed; **same blocker expected** |

**Headline:** **`torch.compile(mode="default")` ≈ 2.1× decode** (15.4 → 32.9 tok/s) and **−38 % TTFT**
(71 → 44 ms), zero accuracy change. Use it for offline GRPO rollouts / batch eval today.

## Why vLLM (and SGLang) don't serve our model yet
vLLM 0.22 **has a native `laguna.py`** — but it implements the **MoE teacher**, not our **dense
student**. Two failure modes, both structural:
1. **Native path** (`vllm/model_executor/models/laguna.py`): expects original-Laguna config fields
   the dense config omits — first `qkv_bias` (injectable via `hf_overrides`), then `decoder_sparse_step`,
   i.e. it wants to build **256-expert MoE** layers.
2. **Transformers backend** (`model_impl="transformers"`): builds `TransformersMoEForCausalLM` and
   then fails to load weights — our checkpoint has `model.layers.N.mlp.routed_dense.{gate,up,down}_proj`
   but vLLM looks for the MoE structure `mlp.experts.w13_weight / w2_weight / gate.weight`:
   > `ValueError: no module named 'model.layers.1.mlp.routed_dense' … available: {experts.w13_weight, experts.w2_weight, gate.weight, shared_experts.*}`

**Root cause:** the config still advertises `num_experts=256` / `model_type=laguna`, so every vLLM path
constructs the **MoE** FFN, which doesn't match our **dense** `routed_dense` weights.

## What we can do (paths to vLLM/SGLang serving)
- **(A) Small vLLM model plugin for `laguna_dense`** *(recommended, ~80–120 LoC).* Copy vLLM's
  `laguna.py`, replace the MoE block (router + `FusedMoE` experts) with **one dense `MergedColumnParallelLinear`
  gate/up + `RowParallelLinear` down (SwiGLU) = `routed_dense`** + the kept shared expert; register it
  as architecture `LagunaDenseForCausalLM`. Keeps attention (GQA 48/64 + SWA + QK-norm + `g_proj`) from
  the native impl. → unlocks paged-attention + continuous batching (the real throughput win).
- **(B) Config surgery so vLLM sees a plain dense model** — drop/zero the MoE fields and expose
  `routed_dense` as a standard MLP via the transformers backend. Lower-effort but fragile (the
  custom attention may still need the plugin).
- **SGLang:** same plugin model applies (per-arch model file); install only after (A)/(B) proves the
  weight mapping. Not worth installing for the generic HF fallback (same MoE-vs-dense mismatch).

## Recommended stack today
| Use case | Recipe |
|---|---|
| **Offline GRPO rollouts / eval (batch)** | HF transformers + **`torch.compile(mode="default")`** (2.1×) |
| **Memory-constrained / on-device** | **HQQ 4-bit** (~1.7 GB) or **torchao int8** (3.21 GB, byte-identical) — see [`QUANTIZATION.md`](QUANTIZATION.md) |
| **High-throughput serving** | **write the `laguna_dense` vLLM plugin (A)** — not available out-of-the-box |

*Reproduce: `scripts` `bench_hf.py` (transformers + compile) and `vllm_try.py` (vLLM loader probe).*
