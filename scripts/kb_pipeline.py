"""Fast KernelBench L1 pipeline — parallel gen + eval on A100 80GB.

Loads model once, generates all 100 kernels in batch, then evaluates
multiple kernels in parallel subprocesses (4-8 concurrent evals).
"""
import json, os, sys, re, subprocess, time, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/data/shinka-kernelbench/KernelBench/src")
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.utils import extract_first_code

HF_CACHE = "/data/cache/huggingface"
WORKER = "/data/kb_eval_worker.py"

# ── Prompt (improved version) ──────────────────────────────────────────
EXAMPLE_INPUT = """import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b

def get_inputs():
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]

def get_init_inputs():
    return []"""

EXAMPLE_OUTPUT = '''import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

elementwise_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { out[idx] = a[idx] + b[idx]; }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""
elementwise_add_cpp_source = "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
elementwise_add = load_inline(name="elementwise_add", cpp_sources=elementwise_add_cpp_source, cuda_sources=elementwise_add_source, functions=["elementwise_add_cuda"], verbose=True)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)'''


def build_prompt(ref_src):
    return """You write custom CUDA kernels embedded in Python using torch.utils.cpp_extension.load_inline to replace PyTorch operators for speedups.

IMPORTANT OUTPUT FORMAT:
- Output a COMPLETE PYTHON file with load_inline (NOT raw C++ with PYBIND11_MODULE)
- Use float* pointers directly (NOT C++ templates or AT_DISPATCH macros)
- Use input.scalar_type() if you need dtype checks (NOT input.type() which is removed)
- ModelNew must have the SAME __init__ signature as Model
- You may keep some PyTorch operators unchanged and only replace specific ones with custom CUDA

Example:

Input architecture:

%s

Optimized output (Python with load_inline):

%s

You are given the following architecture:

%s

Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in a ```python code block. Just output the new model code, no other text, and NO testing code!""" % (EXAMPLE_INPUT, EXAMPLE_OUTPUT, ref_src)


def wrap_cpp_as_python(cpp_code, ref_src):
    if "load_inline" in cpp_code and "class ModelNew" in cpp_code:
        return cpp_code
    if not cpp_code.strip().startswith("#include") and "class ModelNew" in cpp_code:
        return cpp_code
    func_matches = re.findall(r'torch::Tensor\s+(\w+)\s*\([^)]*\)', cpp_code)
    kernel_funcs = set(re.findall(r'__global__\s+void\s+(\w+)', cpp_code))
    cpp_funcs = [f for f in func_matches if f not in kernel_funcs]
    if not cpp_funcs:
        all_funcs = re.findall(r'(?:torch::Tensor|void)\s+(\w+)\s*\(', cpp_code)
        cpp_funcs = [f for f in all_funcs if f not in kernel_funcs and f != "main"]
    if not cpp_funcs:
        cpp_funcs = ["forward"]
    main_func = cpp_funcs[0]
    cuda_src = re.sub(r'PYBIND11_MODULE\s*\([^)]*\)\s*\{[^}]*\}', '', cpp_code).strip()
    cpp_decls = []
    for fn in cpp_funcs:
        sig = re.search(r'(torch::Tensor\s+%s\s*\([^)]*\))' % fn, cpp_code)
        if sig:
            cpp_decls.append(sig.group(1) + ";")
    if not cpp_decls:
        cpp_decls = ["torch::Tensor %s(torch::Tensor input);" % main_func]

    fwd_match = re.search(r'def forward\(self,\s*([^)]*)\)', ref_src)
    args = [a.split(":")[0].strip() for a in fwd_match.group(1).strip().split(",")] if fwd_match else ["x"]
    args_call = ", ".join(args)

    # Parse __init__ args from ref
    init_match = re.search(r'def __init__\(self(?:,\s*([^)]*))?\)', ref_src)
    if init_match and init_match.group(1):
        init_args = init_match.group(1).strip()
    else:
        init_args = ""

    return '''import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """%s"""
cpp_source = """%s"""

custom_ops = load_inline(
    name="custom_ops", cpp_sources=cpp_source, cuda_sources=cuda_source,
    functions=%s, verbose=False, extra_cflags=["-O3"], extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self%s):
        super(ModelNew, self).__init__()

    def forward(self, %s):
        return custom_ops.%s(%s)
''' % (cuda_src, "\\n".join(cpp_decls), cpp_funcs,
       (", " + init_args) if init_args else "",
       args_call, main_func, args_call)


def eval_single(pid, kernel_path, ref_path, result_path):
    """Run eval in subprocess — called by ProcessPoolExecutor."""
    try:
        proc = subprocess.run(
            [sys.executable, WORKER, str(kernel_path), str(ref_path), str(result_path)],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0",
                 "TORCH_EXTENSIONS_DIR": "/data/.torch_extensions",
                 "TMPDIR": "/data/tmp"},
        )
        if os.path.exists(result_path):
            return pid, json.load(open(result_path))
        return pid, {"compiled": False, "correct": False, "speedup": 0.0,
                     "error": "worker failed (rc=%d)" % proc.returncode}
    except subprocess.TimeoutExpired:
        return pid, {"compiled": False, "correct": False, "speedup": 0.0, "error": "timeout 300s"}
    except Exception as e:
        return pid, {"compiled": False, "correct": False, "speedup": 0.0, "error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--skip-gen", action="store_true")
    a = ap.parse_args()

    os.environ["TORCH_EXTENSIONS_DIR"] = "/data/.torch_extensions"
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TMPDIR"] = "/data/tmp"
    os.environ["XDG_CACHE_HOME"] = "/data/cache"

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernels").mkdir(exist_ok=True)

    dataset = construct_kernelbench_dataset(level=1, source="huggingface")
    t_total = time.time()

    # ── PHASE 1: GENERATE ──────────────────────────────────────────────
    if not a.skip_gen:
        print("=== PHASE 1: GENERATE (%s, batch=%d, max_tokens=%d) ===" % (
            a.model, a.batch_size, a.max_new_tokens), flush=True)

        tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, cache_dir=HF_CACHE)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token_id = 9

        model = AutoModelForCausalLM.from_pretrained(
            a.model, trust_remote_code=True, cache_dir=HF_CACHE,
            torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
        )
        model.eval()

        eos_ids = tok.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        stop_ids = list(set(eos_ids + [24]))

        remaining = []
        ref_sources = {}
        for pid in range(1, 101):
            if (out / ("problem_%03d_raw.txt" % pid)).exists():
                continue
            problem = dataset.get_problem_by_id(pid)
            ref_sources[pid] = problem.code
            remaining.append(pid)

        print("  %d to generate (%d cached)" % (len(remaining), 100 - len(remaining)), flush=True)
        t_gen = time.time()
        total_toks = 0

        for i in range(0, len(remaining), a.batch_size):
            batch_pids = remaining[i:i+a.batch_size]
            prompts = []
            for pid in batch_pids:
                user_prompt = build_prompt(ref_sources[pid])
                msgs = [{"role": "user", "content": user_prompt}]
                prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

            inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to("cuda:0")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False,
                                         pad_token_id=tok.pad_token_id, eos_token_id=stop_ids)

            for j, pid in enumerate(batch_pids):
                input_len = inputs["input_ids"][j].shape[0]
                text = tok.decode(outputs[j][input_len:], skip_special_tokens=True)
                total_toks += len(outputs[j]) - input_len
                (out / ("problem_%03d_raw.txt" % pid)).write_text(text)
                kernel = extract_first_code(text, ["python", "cpp"])
                if kernel and kernel.strip().startswith("#include"):
                    kernel = wrap_cpp_as_python(kernel, ref_sources[pid])
                if kernel:
                    (out / "kernels" / ("problem_%03d_kernel.py" % pid)).write_text(kernel)

            elapsed = time.time() - t_gen
            done = min(i + a.batch_size, len(remaining))
            rate = total_toks / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - done) * (elapsed / done) / 60 if done > 0 else 0
            print("  [%3d/%d] %.1f tok/s, ~%.0fm left" % (done, len(remaining), rate, eta), flush=True)

        del model, tok
        torch.cuda.empty_cache()
        print("  Gen done: %d tokens in %.0fs" % (total_toks, time.time() - t_gen), flush=True)
    else:
        print("=== SKIPPING GENERATION (--skip-gen) ===", flush=True)
        ref_sources = {}
        for pid in range(1, 101):
            ref_sources[pid] = dataset.get_problem_by_id(pid).code

    # ── PHASE 2: EVAL (parallel subprocesses) ──────────────────────────
    print("=== PHASE 2: EVAL (%d parallel workers) ===" % a.eval_workers, flush=True)
    t_eval = time.time()

    # Prepare all kernel + ref files
    eval_jobs = []
    for pid in range(1, 101):
        result_path = out / ("problem_%03d_result.json" % pid)
        if result_path.exists():
            continue
        kernel_path = out / "kernels" / ("problem_%03d_kernel.py" % pid)
        if not kernel_path.exists():
            # No kernel — write failure result directly
            r = {"problem_id": pid, "compiled": False, "correct": False, "speedup": 0.0,
                 "kernel_ms": -1, "ref_ms": -1, "error": "no code extracted from model output"}
            json.dump(r, open(result_path, "w"), indent=2)
            continue
        ref_path = out / ("problem_%03d_ref.py" % pid)
        if not ref_path.exists():
            ref_path.write_text(ref_sources.get(pid, dataset.get_problem_by_id(pid).code))
        eval_jobs.append((pid, str(kernel_path), str(ref_path), str(result_path)))

    print("  %d kernels to evaluate (%d cached/skipped)" % (len(eval_jobs), 100 - len(eval_jobs)), flush=True)

    results = []
    done = 0
    # Note: ProcessPoolExecutor for the dispatch, but each job is a subprocess anyway
    # We just run N subprocesses concurrently
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.eval_workers) as executor:
        futures = {executor.submit(eval_single, pid, kp, rp, op): pid
                   for pid, kp, rp, op in eval_jobs}
        for future in concurrent.futures.as_completed(futures):
            pid, r = future.result()
            r["problem_id"] = pid
            results.append(r)
            done += 1
            if done % 10 == 0 or done == len(eval_jobs):
                correct = sum(1 for x in results if x.get("correct"))
                compiled = sum(1 for x in results if x.get("compiled"))
                print("  [%3d/%d] %d compiled, %d correct" % (done, len(eval_jobs), compiled, correct), flush=True)

    # Load cached results too
    for pid in range(1, 101):
        rp = out / ("problem_%03d_result.json" % pid)
        if rp.exists() and not any(r.get("problem_id") == pid for r in results):
            results.append(json.load(open(rp)))

    print("  Eval done in %.0fs" % (time.time() - t_eval), flush=True)

    # ── SUMMARY ────────────────────────────────────────────────────────
    n = len(results)
    compiled = sum(1 for r in results if r.get("compiled"))
    correct = sum(1 for r in results if r.get("correct"))
    speedups = [r["speedup"] for r in results if r.get("correct") and r.get("speedup", 0) > 0]
    faster = sum(1 for s in speedups if s > 1.0)

    print("\n=== RESULTS: %s ===" % a.model, flush=True)
    print("  Compiled: %d/%d (%.1f%%)" % (compiled, n, 100.0*compiled/n), flush=True)
    print("  Correct:  %d/%d (%.1f%%)" % (correct, n, 100.0*correct/n), flush=True)
    print("  Faster:   %d/%d" % (faster, n), flush=True)
    if speedups:
        print("  Avg speedup: %.3fx" % (sum(speedups)/len(speedups)), flush=True)
    print("  Total time: %.0fs" % (time.time() - t_total), flush=True)

    json.dump(results, open(out / "all_results.json", "w"), indent=2)
    json.dump({"model": a.model, "compiled": compiled, "correct": correct,
               "total": n, "faster": faster}, open(out / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
