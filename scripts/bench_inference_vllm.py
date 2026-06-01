import time, json
SNAP="/mnt/data2/hf/models--EvanOLeary--laguna-xs2-dense-k8-cuda-grpo/snapshots/3bfe48919ca635ce1252bcb80ff43e9c3b26681d/"
from vllm import LLM, SamplingParams
# native vllm/models/laguna.py expects original-Laguna config fields the dense config omits
ov={"qkv_bias": False, "attention_bias": False, "mlp_bias": False, "attention_out_bias": False}
print("hf_overrides:", ov, flush=True)
try:
    llm=LLM(model=SNAP, trust_remote_code=True, dtype="bfloat16",
            gpu_memory_utilization=0.15, max_model_len=2048, enforce_eager=True,
            max_num_seqs=1, hf_overrides=ov, model_impl="transformers")
    p="<|im_start|>user\nWrite a CUDA relu kernel.<|im_end|>\n<|im_start|>assistant\n"
    llm.generate([p], SamplingParams(max_tokens=8,temperature=0))
    t1=time.time(); llm.generate([p],SamplingParams(max_tokens=1,temperature=0)); t1=time.time()-t1
    t2=time.time(); o=llm.generate([p],SamplingParams(max_tokens=128,temperature=0)); t2=time.time()-t2
    n=len(o[0].outputs[0].token_ids)
    print(f"[vllm TRANSFORMERS backend] LOADED ✓ | TTFT={t1*1000:.0f} ms | decode={(n-1)/max(t2-t1,1e-6):.1f} tok/s", flush=True)
except Exception as e:
    import traceback
    print("[vllm] FAILED:", type(e).__name__, str(e)[-220:], flush=True)
