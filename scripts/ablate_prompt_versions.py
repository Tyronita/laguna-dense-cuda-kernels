"""A/B/C: quantify what the CUDA-version pin vs full master prompt fixes.
BARE vs VERSION-only vs MASTER system prompt, on SFT-extended, K=4, subprocess-isolated."""
import os, sys, json, subprocess, tempfile, hashlib
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from densify.kernel_reward import extract_code

HERE = os.path.dirname(__file__)
BARE = ("You are a GPU kernel engineer. Write a CUDA kernel for the given PyTorch module. "
        "Define `torch::Tensor forward(torch::Tensor input)` and a PYBIND11_MODULE binding.")
VERSION = BARE + " Target PyTorch 2.7 / CUDA 12.8 — use the current ATen API."
MASTER = (
    "You are an expert CUDA kernel engineer for PyTorch 2.7 / CUDA 12.8 (load_inline). Write ONE correct, "
    "compilable kernel. Rules:\n"
    "- Types: use input.scalar_type() (NOT input.type()); AT_DISPATCH_FLOATING_TYPES with scalar_t.\n"
    "- Sizes: total count = input.numel() (NEVER input.size() with no arg; size(d) needs a dim).\n"
    "- Bounds: int64_t idx=blockIdx.x*blockDim.x+threadIdx.x; if (idx<n){ out[idx]=...; } (don't `return`).\n"
    "- Vectorize ONLY with reinterpret_cast<float4*>(ptr) + a scalar tail; else write a simple scalar kernel.\n"
    "- Pointers: tensor.data_ptr<scalar_t>() (NOT .data<>()). Output: torch::empty_like(input).\n"
    "- Math in float (tanhf/expf/erff). Exactly ONE PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def(\"forward\",&forward);}.\n"
    "Prefer a SIMPLE correct kernel over a fast wrong one. Return only a ```cpp block.")
PROMPTS = [("BARE", BARE), ("VERSION", VERSION), ("MASTER", MASTER)]
OPS = ["relu", "tanh", "sigmoid", "gelu", "softplus", "silu"]
BODY = {"relu": "torch.relu(x)", "tanh": "torch.tanh(x)", "sigmoid": "torch.sigmoid(x)",
        "gelu": "torch.nn.functional.gelu(x)", "softplus": "torch.nn.functional.softplus(x)",
        "silu": "torch.nn.functional.silu(x)"}
K = 4


def gen_k(model, tok, sysp, op):
    user = f"Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\nimport torch\nimport torch.nn as nn\nclass Model(nn.Module):\n    def forward(self, x):\n        return {BODY[op]}\n```"
    s = tok.apply_chat_template([{"role": "system", "content": sysp}, {"role": "user", "content": user}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=1024, do_sample=True, temperature=0.6, top_k=20, num_return_sequences=K, pad_token_id=9)
    return [extract_code(tok.decode(o[ids.shape[-1]:], skip_special_tokens=True)) for o in out]


def iso(code, op, tag):
    name = f"{op}_{tag}_{hashlib.md5(code.encode()).hexdigest()[:6]}"
    with tempfile.TemporaryDirectory() as d:
        i, o = f"{d}/i.json", f"{d}/o.json"
        json.dump({"code": code, "dsl": "CUDA", "op": op, "name": name}, open(i, "w"))
        env = dict(os.environ, CUDA_HOME="/usr/local/cuda", PATH="/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
                   PYTHONPATH=os.path.join(HERE, "..", "src"), TORCH_CUDA_ARCH_LIST="9.0", TORCH_EXTENSIONS_DIR=f"{d}/ext")
        try:
            subprocess.run([sys.executable, f"{HERE}/eval_worker.py", "--in", i, "--out", o], env=env, timeout=160, capture_output=True)
            return json.load(open(o)) if os.path.exists(o) else {"compiled": False, "correct": False}
        except subprocess.TimeoutExpired:
            return {"compiled": False, "correct": False}


def main():
    ck = "runs/sft/kernel_cuda_sft_v2/checkpoint-final"
    tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ck, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    out = {}
    for label, sysp in PROMPTS:
        comp = corr = 0
        for op in OPS:
            codes = gen_k(model, tok, sysp, op)
            with ThreadPoolExecutor(max_workers=K) as ex:
                rs = list(ex.map(lambda ck_: iso(ck_[1], op, f"{label}{ck_[0]}"), list(enumerate(codes))))
            c = max(int(r.get("compiled", False)) for r in rs); o = max(int(r.get("correct", False)) for r in rs)
            comp += c; corr += o
            print(f"[{label}] {op:9} compile@{K}={c} correct@{K}={o}", flush=True)
        out[label] = {"compile": comp, "correct": corr, "n": len(OPS)}
        print(f"[{label}] TOTAL compile {comp}/{len(OPS)}  correct {corr}/{len(OPS)}\n", flush=True)
    os.makedirs("runs/eval", exist_ok=True)
    json.dump(out, open("runs/eval/prompt_version_ablation.json", "w"), indent=2)
    print("=" * 50)
    for label, _ in PROMPTS:
        s = out[label]; print(f"{label:8} compile {s['compile']}/{s['n']}  correct {s['correct']}/{s['n']}")


if __name__ == "__main__":
    main()
