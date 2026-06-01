import os, time
os.environ["VLLM_LOGGING_LEVEL"]="WARNING"; os.environ["VLLM_USE_FLASHINFER_SAMPLER"]="0"
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
SNAP="/mnt/data2/hf/models--EvanOLeary--laguna-xs2-dense-k8-cuda-grpo/snapshots/3bfe48919ca635ce1252bcb80ff43e9c3b26681d/"
tok=AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
def mk(op): return tok.apply_chat_template(
   [{"role":"system","content":"You are an expert GPU kernel engineer."},
    {"role":"user","content":f"Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\nclass Model(nn.Module):\n    def forward(self,x): return torch.{op}(x)\n```"}],
   add_generation_prompt=True, tokenize=False, enable_thinking=False)
ops=["relu","sigmoid","tanh","gelu","abs","exp","log","sqrt"]
ov={"decoder_sparse_step":1,"mlp_only_layers":[0],"qkv_bias":False,"attention_bias":False,"attention_out_bias":False,"mlp_bias":False}
llm=LLM(model=SNAP, trust_remote_code=True, dtype="bfloat16", gpu_memory_utilization=0.55,
        max_model_len=2048, enforce_eager=True, max_num_seqs=64, hf_overrides=ov)
sp=SamplingParams(max_tokens=128, temperature=0)
for B in (1, 8, 32, 64):
    prompts=[mk(ops[i%len(ops)]) for i in range(B)]
    llm.generate(prompts[:1], SamplingParams(max_tokens=4,temperature=0))  # warm
    t=time.time(); outs=llm.generate(prompts, sp); dt=time.time()-t
    toks=sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[batch={B:3d}] {dt:5.1f}s | {toks} out-toks | AGG {toks/dt:6.1f} tok/s | per-req {toks/dt/B:5.1f} tok/s", flush=True)
