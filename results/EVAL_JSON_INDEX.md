# Raw eval JSONs (machine-readable results)

## head_to_head.json — ours (3.0B SFT) vs Laguna teacher (33.4B), same 6 CUDA ops
- **ours**: 3.0B, 32.1 tok/s, fast_0 0/6 (k=1)
- **teacher**: 33.4B, 25.4 tok/s, fast_0 4/6 (k=1)

## eval_10ops.json — SFT model, 10 ops x {CUDA,Triton}, pass@3, isolated
- **CUDA**: correct pass@3 2/10 | eval avg 33.9s
- **Triton**: correct pass@3 0/10 | eval avg 2.0s

## prompt_version_ablation.json — SFT-ext, BARE vs VERSION vs MASTER (6 ops, K=4)
- **BARE**: compile 5/6 correct 2/6
- **VERSION**: compile 6/6 correct 2/6
- **MASTER**: compile 2/6 correct 0/6
