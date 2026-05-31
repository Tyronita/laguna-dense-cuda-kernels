import os, time
os.environ["VLLM_LOGGING_LEVEL"]="WARNING"; os.environ["VLLM_USE_FLASHINFER_SAMPLER"]="0"
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
SNAP="/mnt/data2/hf/models--EvanOLeary--laguna-xs2-dense-k8-cuda-grpo/snapshots/3bfe48919ca635ce1252bcb80ff43e9c3b26681d/"
tok=AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
msgs=[{"role":"system","content":"You are an expert GPU kernel engineer. Convert PyTorch modules into correct, optimized CUDA kernels."},
      {"role":"user","content":"Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\nclass Model(nn.Module):\n    def forward(self,x): return torch.relu(x)\n```"}]
prompt=tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
ov={"decoder_sparse_step":1,"mlp_only_layers":[0],"qkv_bias":False,"attention_bias":False,"attention_out_bias":False,"mlp_bias":False}
llm=LLM(model=SNAP, trust_remote_code=True, dtype="bfloat16", gpu_memory_utilization=0.3,
        max_model_len=2048, enforce_eager=True, max_num_seqs=1, hf_overrides=ov)
llm.generate([prompt], SamplingParams(max_tokens=8,temperature=0))
t1=time.time(); llm.generate([prompt],SamplingParams(max_tokens=1,temperature=0)); t1=time.time()-t1
t2=time.time(); o=llm.generate([prompt],SamplingParams(max_tokens=200,temperature=0)); t2=time.time()-t2
n=len(o[0].outputs[0].token_ids)
print(f"[vLLM laguna_dense] TTFT={t1*1000:.0f} ms | decode={(n-1)/max(t2-t1,1e-6):.1f} tok/s | gen={n}", flush=True)
print("[sample]", repr(o[0].outputs[0].text[:280]), flush=True)
