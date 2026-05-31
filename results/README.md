# Results

Evaluation results organized by experiment. Each subfolder contains generated kernels, evaluation metrics, and a README describing the setup.

## Experiments

| # | Experiment | Model(s) | Problems | Key metric |
|---|---|---|---|---|
| **01** | [Head-to-Head](01_head_to_head/) | SFT vs Teacher (33B MoE) | 6 elementwise ops | 26% faster decode, 11x smaller |
| **02** | [Smoke Test 10 Ops](02_smoke_test_10ops/) | DPO (pass@3) | 10 elementwise ops | ReLU/Tanh ~70% correct |
| **03** | [KernelBench L1](03_kernelbench_l1/) | All 5 variants | 100 KB-L1 problems | Full benchmark (in progress) |

## Model lineage & naming

```
poolside/Laguna-XS.2 (33B MoE)
  → laguna-xs2-dense-k8-kernelmix (pretrain V2)
    → laguna-xs2-dense-k8-cuda-sft          ← SFT (level_1+2, 400 steps)
      → laguna-xs2-dense-k8-cuda-sft-v2     ← SFT-extended (level_1+2+3, +500 steps)
        ├── cuda-grpo  = GRPO-offline        ← Dr.GRPO on Sakana dataset rewards (off-policy)
        ├── cuda-rft   = GRPO-online         ← Dr.GRPO with live compilation (on-policy)
        └── cuda-dpo   = DPO                 ← Preference optimization on Sakana pairs
```

**Note on HF names**: `cuda-grpo` is actually offline (dataset rewards), `cuda-rft` is actually online GRPO (live compilation). See [RESULTS.md](RESULTS.md) for the full explanation.

## Hardware
| Machine | GPU | Use |
|---|---|---|
| a100-bench (Azure) | A100 80GB (Ampere) | Generation + eval timing benchmarks |
| evan (Azure) | T4 16GB (Turing) | Parallel generation |
