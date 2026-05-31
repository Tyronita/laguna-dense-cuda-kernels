import os, time, statistics, torch, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HOME","/mnt/data2/hf")
from transformers import AutoModelForCausalLM, AutoTokenizer
REPO="EvanOLeary/laguna-xs2-dense-k8-cuda-grpo"
SYS="You are an expert GPU kernel engineer. Convert PyTorch modules into correct, optimized CUDA kernels."
USER="Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\nclass Model(nn.Module):\n    def forward(self, x):\n        return torch.relu(x)\n```"
tok=AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True, dtype=torch.bfloat16, device_map={"":"cuda"}, attn_implementation="sdpa").eval()
model.config.use_cache=True
s=tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":USER}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
ids=tok(s, return_tensors="pt").input_ids.to("cuda")
print(f"[prompt] {ids.shape[1]} tokens", flush=True)

def bench(label, gen_tokens=128, trials=3):
    # warmup
    with torch.no_grad(): model.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id or 9)
    torch.cuda.synchronize()
    ttfts, thrus = [], []
    for _ in range(trials):
        torch.cuda.synchronize(); t0=time.time()
        with torch.no_grad(): model.generate(ids, max_new_tokens=1, do_sample=False, pad_token_id=tok.pad_token_id or 9)
        torch.cuda.synchronize(); ttft=time.time()-t0
        torch.cuda.synchronize(); t0=time.time()
        with torch.no_grad(): out=model.generate(ids, max_new_tokens=gen_tokens, do_sample=False, pad_token_id=tok.pad_token_id or 9)
        torch.cuda.synchronize(); dt=time.time()-t0
        n=out.shape[1]-ids.shape[1]
        ttfts.append(ttft); thrus.append((n-1)/max(dt-ttft,1e-6))
    print(f"[{label}] TTFT={statistics.median(ttfts)*1000:.0f} ms | decode={statistics.median(thrus):.1f} tok/s | "
          f"end-to-end {gen_tokens}tok={statistics.median([gen_tokens/ (statistics.median(ttfts)+gen_tokens/statistics.median(thrus)) for _ in [0]]):.1f} tok/s", flush=True)
    return statistics.median(ttfts), statistics.median(thrus)

bench("HF bf16 eager")
print("[torch.compile] compiling (first call slow)...", flush=True)
try:
    model.forward=torch.compile(model.forward, mode="default", fullgraph=False)
    bench("HF bf16 + torch.compile")
except Exception as e:
    print("[torch.compile] FAILED:", type(e).__name__, str(e)[:160], flush=True)
print(f"[mem] peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
