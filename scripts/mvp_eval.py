"""MVP verbose eval: all variants + Laguna teacher, 6 easy tasks, version-pin prompt, K=4.
PRINTS every input, generated kernel, and FULL compile error. Saves runs/eval/mvp_full.md + json."""
import os, sys, json, subprocess, tempfile, hashlib
from concurrent.futures import ThreadPoolExecutor
import pybind11, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from densify.kernel_reward import extract_code

HERE = os.path.dirname(__file__)
PYBIND = pybind11.get_include()
SYS = ("You are a GPU kernel engineer. Write a CUDA kernel for the given PyTorch module. "
       "Define `torch::Tensor forward(torch::Tensor input)` and a PYBIND11_MODULE binding. "
       "Target PyTorch 2.7 / CUDA 12.8 — use the current ATen API.")
OPS = ["relu", "tanh", "sigmoid", "gelu", "abs", "silu"]
BODY = {"relu": "torch.relu(x)", "tanh": "torch.tanh(x)", "sigmoid": "torch.sigmoid(x)",
        "gelu": "torch.nn.functional.gelu(x)", "abs": "torch.abs(x)", "silu": "torch.nn.functional.silu(x)"}
MODELS = [
    ("SFT", "runs/sft/kernel_cuda_sft/checkpoint-final"),
    ("SFT-ext", "runs/sft/kernel_cuda_sft_v2/checkpoint-final"),
    ("GRPO", "runs/rft/grpo/checkpoint-final"),
    ("DPO", "runs/rft/dpo/checkpoint-final"),
    ("Laguna-teacher", "poolside/Laguna-XS.2"),
]
K = 4


def prompt(op):
    return f"Convert this PyTorch module into an optimized CUDA kernel:\n\n```python\nimport torch\nimport torch.nn as nn\nclass Model(nn.Module):\n    def forward(self, x):\n        return {BODY[op]}\n```"


def gen_k(model, tok, op):
    s = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": prompt(op)}],
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
                   PYTHONPATH=os.path.join(HERE, "..", "src"), TORCH_CUDA_ARCH_LIST="9.0",
                   TORCH_EXTENSIONS_DIR=f"{d}/ext", CPLUS_INCLUDE_PATH=PYBIND + ":" + os.environ.get("CPLUS_INCLUDE_PATH", ""))
        try:
            p = subprocess.run([sys.executable, f"{HERE}/eval_worker.py", "--in", i, "--out", o], env=env, timeout=160, capture_output=True, text=True)
            r = json.load(open(o)) if os.path.exists(o) else {"compiled": False, "correct": False, "error": "worker-crash"}
            r["full_error"] = (r.get("error") or "") + "\n" + p.stderr[-1500:] if not r.get("compiled") else ""
            return r
        except subprocess.TimeoutExpired:
            return {"compiled": False, "correct": False, "error": "timeout", "full_error": "timeout"}


def main():
    md = ["# MVP verbose eval — all variants + Laguna teacher (6 tasks, K=4, version-pin prompt, isolated)\n",
          f"**System prompt:** `{SYS}`\n"]
    summary = {}
    for tag, path in MODELS:
        if path != "poolside/Laguna-XS.2" and not os.path.exists(os.path.join(path, "model.safetensors")):
            print(f"[skip {tag}] missing", flush=True); continue
        print(f"\n{'='*70}\n### LOADING {tag} ({path})\n{'='*70}", flush=True)
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
        md.append(f"\n## {tag}\n")
        comp = corr = 0; speeds = []
        for op in OPS:
            print(f"\n----- [{tag}] TASK: {op}  ({BODY[op]}) -----", flush=True)
            print(f"INPUT PROMPT:\n{prompt(op)}", flush=True)
            codes = gen_k(model, tok, op)
            with ThreadPoolExecutor(max_workers=K) as ex:
                rs = list(ex.map(lambda ck: iso(ck[1], op, f"{tag}{ck[0]}"), list(enumerate(codes))))
            c = max(int(r.get("compiled", False)) for r in rs); o = max(int(r.get("correct", False)) for r in rs)
            comp += c; corr += o
            best = [(r.get("speedup") or 0) for r in rs if r.get("correct")]
            if best: speeds.append(max(best))
            md.append(f"\n### {tag} · {op} → compile@4={c} correct@4={o}\n")
            for k, (code, r) in enumerate(zip(codes, rs)):
                status = "✅correct" if r.get("correct") else ("⚠️compiled-wrong" if r.get("compiled") else "❌compile-fail")
                print(f"  [{tag}/{op}] sample {k}: {status} {('sp='+str(round(r['speedup'],2))) if r.get('correct') and r.get('speedup') else ''}", flush=True)
                if not r.get("compiled"):
                    err = (r.get("full_error") or "")[-400:]
                    print(f"      COMPILE ERROR: ...{err.strip()[-300:]}", flush=True)
                md.append(f"**sample {k}: {status}**\n```cpp\n{code[:1500]}\n```\n")
                if not r.get("compiled"):
                    md.append(f"compile error:\n```\n{(r.get('full_error') or '')[-800:]}\n```\n")
        summary[tag] = {"compile": comp, "correct": corr, "n": len(OPS),
                        "mean_best_speedup": round(sum(speeds)/len(speeds), 3) if speeds else None}
        print(f"\n>>> [{tag}] compile {comp}/{len(OPS)}  correct {corr}/{len(OPS)}  speedup {summary[tag]['mean_best_speedup']}", flush=True)
        del model; torch.cuda.empty_cache()
    md.insert(2, "\n## Summary\n| Model | compile@4 | correct@4 | mean speedup |\n|---|---|---|---|\n" +
              "\n".join(f"| {t} | {s['compile']}/{s['n']} | {s['correct']}/{s['n']} | {s['mean_best_speedup'] or '—'} |" for t, s in summary.items()) + "\n")
    os.makedirs("runs/eval", exist_ok=True)
    open("runs/eval/mvp_full.md", "w").write("\n".join(md))
    json.dump(summary, open("runs/eval/mvp_summary.json", "w"), indent=2)
    print("\n" + "=" * 50 + "\nFINAL:")
    for t, s in summary.items():
        print(f"  {t:16} compile {s['compile']}/{s['n']}  correct {s['correct']}/{s['n']}  speedup {s['mean_best_speedup']}")
    print("[saved runs/eval/mvp_full.md + mvp_summary.json]")


if __name__ == "__main__":
    main()
