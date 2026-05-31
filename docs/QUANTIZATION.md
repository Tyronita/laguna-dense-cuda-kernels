# Laguna-Dense Quantization Recipes

Reference for quantizing the Laguna-XS.2-dense family (or any custom-architecture HF model with `trust_remote_code=True`).
All recipes tested in `/data/laguna/.venv` (torch 2.12 + cu13 + transformers 5.9 on A100 80 GB).

## TL;DR — which recipe to use

| You want | Use | Script | Result |
|---|---|---|---|
| Smallest viable + no headaches | **torchao int8** | `quantize_torchao.py --mode int8` | 5.99 GB → 3.21 GB (54%), byte-identical on greedy decode |
| Smaller (4-bit), no calibration | **HQQ 4-bit** | `quantize_torchao.py` + HQQ snippet below | 5.99 GB → ~1.7 GB (28%) |
| Best 4-bit quality | AWQ | not yet implemented here | needs ~10 min calibration + architecture patch |
| Long-context serving | torchao + OSCAR KV | research path | KV cache int2 + weights int4/8 |

## What does NOT work in this env

| Recipe | Why it fails |
|---|---|
| **bitsandbytes 0.49.2** | CUDA kernel symbols don't resolve under torch 2.12 → `RuntimeError: Bnb4bitQuantize` per-tensor. Either downgrade torch to 2.5 or wait for bnb 0.50. |
| **torchao Int4WeightOnly** (0.17) | Requires `mslk >= 1.0.0` kernel extension — not pip-installable. |
| **NVFP4** | Native compute requires Blackwell (B100/B200). A100 Ampere has no FP4 tensor cores. |
| **FP8 (TransformerEngine)** | Hopper-only. A100 has no native FP8. |

## Recipe 1 — torchao Int8 weight-only (RECOMMENDED for first pass)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torchao.quantization import quantize_, Int8WeightOnlyConfig

repo = "EvanOLeary/laguna-xs2-dense-k8-cuda-sft"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo, trust_remote_code=True, dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="sdpa",
)
quantize_(model, Int8WeightOnlyConfig())  # 0.4 s on A100, lossless on greedy

# save
model.save_pretrained("./out_int8", safe_serialization=False)  # .bin, not safetensors
tok.save_pretrained("./out_int8")
# Also copy modeling_*.py + configuration_*.py + chat_template.jinja from source repo
```

**Cost:** −46% VRAM, −34% tok/s without `torch.compile`. Compile recovers most of the throughput.

## Recipe 2 — HQQ 4-bit

```python
from hqq.core.quantize import BaseQuantizeConfig
from hqq.models.hf.base import AutoHQQHFModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "EvanOLeary/laguna-xs2-dense-k8-cuda-sft"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo, trust_remote_code=True, dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="sdpa",
)

cfg = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)
AutoHQQHFModel.quantize_model(model, quant_config=cfg,
                               compute_dtype=torch.bfloat16, device="cuda")
# ~4 s on A100

# save
AutoHQQHFModel.save_quantized(model, save_dir="./out_hqq_int4")
tok.save_pretrained("./out_hqq_int4")
```

**Load:**
```python
from hqq.models.hf.base import AutoHQQHFModel
model = AutoHQQHFModel.from_quantized("./out_hqq_int4",
        compute_dtype=torch.bfloat16, device="cuda")
```

## Recipe 3 — bf16 vs INT8 side-by-side diff

`/data/laguna/workspace/relu_compare.py` — loads bf16, generates, frees, loads bf16 again, quantizes,
generates same prompt, prints both + IDENTICAL/DIFFERENT verdict. Useful sanity check for any new
quant config.

## Pushing to HF

Saved as `/data/laguna/workspace/push_quantized.py`. Pattern:
1. `HfApi().create_repo(repo_id, exist_ok=True, repo_type="model")`
2. Write a `README.md` with frontmatter `base_model:` + quantization details
3. `api.upload_folder(folder_path=..., repo_id=..., repo_type="model")`
4. **Critical:** copy `modeling_*.py` + `configuration_*.py` + `chat_template.jinja` from the bf16
   source repo into the quantized dir before upload — otherwise `trust_remote_code` loaders break.

## Live model repos using these recipes

| Repo | Format | Size |
|---|---|---:|
| `EvanOLeary/laguna-xs2-dense-k8-cuda-sft` | bf16 (source) | 5.99 GB |
| `EvanOLeary/laguna-xs2-dense-k8-cuda-sft-int8` | torchao Int8 | 3.21 GB |
| `EvanOLeary/laguna-xs2-dense-k8-cuda-sft-int4-hqq` | HQQ 4-bit | ~1.7 GB |
