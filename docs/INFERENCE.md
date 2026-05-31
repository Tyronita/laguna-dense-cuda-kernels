# Inference optimization — measured (tok/s + TTFT) & serving backends

**Model:** `EvanOLeary/laguna-xs2-dense-k8-cuda-grpo` (~3.0 B dense). **HW:** A100 80 GB.
**Greedy, 77-token prompt.** `decode` = steady-state tok/s; `TTFT` = prefill + first token.

## Results

| Backend | TTFT | single-seq decode | batched throughput | correctness |
|---|---|---|---|---|
| HF transformers, bf16 **eager** | 71 ms | 15.4 tok/s | — | ✅ valid CUDA |
| HF transformers **+ `torch.compile`** (`mode="default"`) | **44 ms** | **32.9 tok/s (2.1×)** | — | ✅ |
| **vLLM 0.22 (dense plugin, `enforce_eager`)** | 51 ms | 21.6 tok/s | **see below** | ✅ valid CUDA |

`torch.compile(mode="default")` ✅ = **2.1×** single-seq (`max-autotune` ❌ — CUDA-graphs clash with `generate`).

### vLLM batched throughput (the real win — continuous batching)
| batch | aggregate tok/s | per-req | vs HF eager |
|---|---|---|---|
| 1 | 21.4 | 21.4 | 1.4× |
| 8 | 161 | 20.2 | 10× |
| 32 | 621 | 19.4 | 40× |
| **64** | **1227** | 19.2 | **~80×** |

**Takeaways:** for **single sequences**, HF + `torch.compile` (32.9 tok/s) is fastest. For
**throughput** (GRPO rollouts = G samples × many prompts; serving), **vLLM wins decisively** —
**~1227 tok/s at batch 64** (≈80× HF eager, ≈37× HF-compile). Use HF+compile for one-off generation,
vLLM for rollout/eval fleets.

## Getting vLLM to serve the DENSE student
vLLM 0.22 ships a native `laguna.py` — but it's the **MoE teacher** (builds 256-expert FFNs and
can't load our `routed_dense` weights). The fix is a **~20-line, `model_type`-gated** addition that
reuses vLLM's native `LagunaMLP` and **leaves OG Laguna untouched** (`model_type=="laguna"` keeps the
real `LagunaMoE`):

```python
# in vllm/model_executor/models/laguna.py
class LagunaDenseFFN(nn.Module):                       # dense replacement of the routed MoE block
    def __init__(self, config, quant_config=None, prefix="", enable_eplb=False):
        super().__init__()
        h = config.hidden_size
        routed = config.num_experts_per_tok * config.moe_intermediate_size   # 8*512 = 4096
        self.routed_dense   = LagunaMLP(h, routed, config.hidden_act, quant_config, prefix=f"{prefix}.routed_dense")
        self.shared_experts = LagunaMLP(h, config.shared_expert_intermediate_size, config.hidden_act, quant_config, prefix=f"{prefix}.shared_experts")
        self.scale = float(getattr(config, "moe_routed_scaling_factor", 1.0))  # 2.5
    def forward(self, x):
        return self.routed_dense(x) * self.scale + self.shared_experts(x)

# in LagunaDecoderLayer.__init__, BEFORE the existing MoE branch:
if self.is_moe_layer and getattr(config, "model_type", "") == "laguna_dense":
    self.mlp = LagunaDenseFFN(config=config, quant_config=quant_config, prefix=f"{prefix}.mlp")
elif self.is_moe_layer:
    self.mlp = LagunaMoE(...)        # OG Laguna unchanged
```

Run it with:
```python
LLM(model="EvanOLeary/laguna-xs2-dense-k8-cuda-grpo", trust_remote_code=True,
    hf_overrides={"mlp_only_layers":[0], "decoder_sparse_step":1, "qkv_bias":False},
    enforce_eager=True)   # native attention (GQA 48/64 + SWA + QK-norm + g_proj) is reused as-is
# env: VLLM_USE_FLASHINFER_SAMPLER=0  (the flashinfer sampler JIT-fails on this CUDA stack)
```
**Two gotchas that cost us:** ① the **chat prompt must use `tokenizer.apply_chat_template`** (a
hand-written template → degenerate output); ② **flashinfer sampler** fails to JIT-compile → disable it.

> ⚠️ **Do this in an isolated venv** (`/mnt/data2/vllm_dense_venv`) — the shared `kb-eval-venv` vLLM
> is used by another workstream for the **OG Laguna MoE**; never patch its `laguna.py` globally. The
> `model_type` guard makes the change safe to **upstream** properly.

**SGLang:** same one-file model approach (a `routed_dense + shared` block + `load_weights`); not yet done.

## Recommended stack
| Use case | Recipe |
|---|---|
| One-off generation | HF + **`torch.compile(mode="default")`** (2.1×, 32.9 tok/s) |
| **GRPO rollouts / eval / serving** | **vLLM + the `laguna_dense` patch** (~1227 tok/s @ batch 64) |
| Memory-constrained / on-device | **HQQ 4-bit** (~1.7 GB) / **torchao int8** (3.21 GB) — [`QUANTIZATION.md`](QUANTIZATION.md) |

*Reproduce: `scripts/bench_inference_hf.py` (HF + compile), `scripts/bench_vllm_dense.py` (single-seq), `scripts/bench_vllm_batch.py` (throughput).*
