"""
KernelBench evaluation for laguna-xs2 models (GRPO, GRPO, DPO).

Uses HuggingFace transformers with proper chat template + KernelBench eval.
Target: A100 80GB on Azure VM.
"""
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/data/shinka-kernelbench/KernelBench/src")
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string
from kernelbench.timing import measure_ref_program_time
from kernelbench.utils import extract_first_code, set_gpu_arch
from kernelbench.kernel_static_checker import validate_kernel_static

# ── Config ──────────────────────────────────────────────────────────────
MODELS = [
    "EvanOLeary/laguna-xs2-dense-k8-cuda-grpo",
    "EvanOLeary/laguna-xs2-dense-k8-cuda-rft",
    "EvanOLeary/laguna-xs2-dense-k8-cuda-dpo",
]

LEVEL = 1
NUM_PROBLEMS = 100
NUM_CORRECT_TRIALS = 5
NUM_PERF_TRIALS = 100
GPU_ARCH = ["Ampere"]
PRECISION = "fp32"
BACKEND = "cuda"
TIMING_METHOD = "cuda_event"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.0

OUTPUT_DIR = Path("/data/kernelbench-laguna-eval")
HF_CACHE = "/data/cache/huggingface"

# ── One-shot example ────────────────────────────────────────────────────
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


def build_user_prompt(ref_arch_src: str) -> str:
    return f"""You write custom CUDA operators to replace the pytorch operators in the given architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA operators and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.

Here's an example to show you the syntax of inline embedding custom CUDA operators in PyTorch:

Input architecture:

{EXAMPLE_INPUT}

Optimized with CUDA operators:

{EXAMPLE_OUTPUT}

You are given the following architecture:

{ref_arch_src}

Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!"""


def wrap_cpp_as_python(cpp_code: str, ref_src: str) -> str:
    """Convert raw C++ CUDA code to Python load_inline format for KernelBench eval.

    Parses the C++ to find exported functions and creates a ModelNew class that
    calls the compiled extension.
    """
    if "load_inline" in cpp_code and "class ModelNew" in cpp_code:
        return cpp_code  # Already in correct format

    if not cpp_code.strip().startswith("#include") and "class ModelNew" in cpp_code:
        return cpp_code  # Already Python

    # Extract function name from PYBIND11_MODULE or from function signatures
    func_name = "forward"
    pybind_match = re.search(r'm\.def\("(\w+)"', cpp_code)
    if pybind_match:
        func_name = pybind_match.group(1)

    # Find C++ function declarations (torch::Tensor return type)
    func_matches = re.findall(r'torch::Tensor\s+(\w+)\s*\([^)]*\)', cpp_code)
    # Filter out kernel functions (those with __global__)
    kernel_funcs = set(re.findall(r'__global__\s+void\s+(\w+)', cpp_code))
    cpp_funcs = [f for f in func_matches if f not in kernel_funcs]

    if not cpp_funcs:
        # Try looking for any non-kernel function
        all_funcs = re.findall(r'(?:torch::Tensor|void)\s+(\w+)\s*\(', cpp_code)
        cpp_funcs = [f for f in all_funcs if f not in kernel_funcs and f != "main"]

    if not cpp_funcs:
        cpp_funcs = ["forward"]

    main_func = cpp_funcs[0]

    # Remove PYBIND11_MODULE block (load_inline handles this)
    cuda_src = re.sub(r'PYBIND11_MODULE\s*\([^)]*\)\s*\{[^}]*\}', '', cpp_code).strip()

    # Build cpp_source declarations
    # Extract full function signatures for cpp declarations
    cpp_decls = []
    for fn in cpp_funcs:
        sig_match = re.search(rf'(torch::Tensor\s+{fn}\s*\([^)]*\))', cpp_code)
        if sig_match:
            cpp_decls.append(sig_match.group(1) + ";")

    if not cpp_decls:
        cpp_decls = [f"torch::Tensor {main_func}(torch::Tensor input);"]

    cpp_source_str = "\\n".join(cpp_decls)

    # Determine forward args from ref_src
    # Parse the original Model.forward signature
    fwd_match = re.search(r'def forward\(self,\s*([^)]*)\)', ref_src)
    if fwd_match:
        args_str = fwd_match.group(1).strip()
        # Remove type annotations
        args = [a.split(":")[0].strip() for a in args_str.split(",")]
    else:
        args = ["x"]

    args_call = ", ".join(args)

    python_code = f'''import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """{cuda_src}"""

cpp_source = """{cpp_source_str}"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions={cpp_funcs},
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, {args_call}):
        return custom_ops.{main_func}({args_call})
'''
    return python_code


def generate_kernel(model, tokenizer, user_prompt, device):
    """Generate a kernel using chat template."""
    messages = [{"role": "user", "content": user_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors='pt').to(device)

    # Stop at </assistant> token (id=24) or EOS
    eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    stop_ids = list(set(eos_ids + [24]))  # 24 = </assistant>

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_ids,
        )

    new_toks = out.shape[1] - inputs["input_ids"].shape[1]
    output_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return output_text, new_toks


def evaluate_kernel(kernel_src: str, ref_src: str, device: torch.device) -> dict:
    """Evaluate a single kernel for correctness and performance."""
    try:
        eval_result = eval_kernel_against_ref(
            original_model_src=ref_src,
            custom_model_src=kernel_src,
            measure_performance=True,
            timing_method=TIMING_METHOD,
            verbose=False,
            num_correct_trials=NUM_CORRECT_TRIALS,
            num_perf_trials=NUM_PERF_TRIALS,
            device=device,
            backend=BACKEND,
            precision=get_torch_dtype_from_string(PRECISION),
        )
    except Exception as e:
        return {
            "compiled": False, "correct": False, "speedup": 0.0,
            "kernel_ms": -1, "ref_ms": -1, "error": str(e)[:500],
        }

    if eval_result is None:
        return {
            "compiled": False, "correct": False, "speedup": 0.0,
            "kernel_ms": -1, "ref_ms": -1, "error": "eval returned None",
        }

    compiled = eval_result.compiled
    correct = eval_result.correctness
    kern_ms = eval_result.runtime if eval_result.runtime and eval_result.runtime > 0 else -1

    error = ""
    if not compiled:
        error = str(eval_result.metadata.get("compilation_error", "compile fail"))[:500]
    elif not correct:
        error = "correctness check failed"

    return {
        "compiled": compiled,
        "correct": correct,
        "kernel_ms": kern_ms,
        "ref_ms": -1,
        "speedup": 0.0,
        "error": error,
    }


def run_eval_for_model(model_name: str, dataset, device: torch.device) -> dict:
    """Generate + evaluate all level-1 problems for a single model."""
    short_name = model_name.split("/")[-1]
    model_dir = OUTPUT_DIR / short_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}", flush=True)

    # Load model
    print(f"Loading {model_name}...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=HF_CACHE
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=HF_CACHE,
        torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

    # Generate kernels one at a time
    problem_ids = list(range(1, NUM_PROBLEMS + 1))
    kernels = []
    ref_sources = []
    total_gen_tokens = 0
    gen_start = time.time()

    print(f"Generating kernels for {NUM_PROBLEMS} problems...", flush=True)
    for i, pid in enumerate(problem_ids):
        problem = dataset.get_problem_by_id(pid)
        ref_src = problem.code
        ref_sources.append(ref_src)

        # Check for existing raw output (resume support)
        raw_path = model_dir / f"problem_{pid:03d}_raw.txt"
        if raw_path.exists():
            raw_text = raw_path.read_text()
            n_toks = 0
        else:
            user_prompt = build_user_prompt(ref_src)
            raw_text, n_toks = generate_kernel(model, tokenizer, user_prompt, device)
            raw_path.write_text(raw_text)
        total_gen_tokens += n_toks

        kernel = extract_first_code(raw_text, ["python", "cpp"])
        # Wrap raw C++ in Python load_inline if needed
        if kernel and kernel.strip().startswith("#include"):
            kernel = wrap_cpp_as_python(kernel, ref_src)
        kernels.append(kernel)

        if kernel:
            (model_dir / f"problem_{pid:03d}_kernel.py").write_text(kernel)

        status = "OK" if kernel else "NO_CODE"
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - gen_start
            rate = total_gen_tokens / elapsed if elapsed > 0 else 0
            print(f"  [{i+1:3d}/{NUM_PROBLEMS}] {status} | {n_toks} toks | {rate:.1f} tok/s avg", flush=True)

    gen_time = time.time() - gen_start
    print(f"Generation done: {total_gen_tokens} tokens in {gen_time:.1f}s ({total_gen_tokens/gen_time:.1f} tok/s)", flush=True)

    # Free model memory
    del model, tokenizer
    torch.cuda.empty_cache()
    import gc; gc.collect()
    time.sleep(2)

    # Evaluate each kernel using subprocess isolation (prevents CUDA crashes from propagating)
    import subprocess
    results = []
    print(f"\nEvaluating {len(kernels)} kernels (subprocess-isolated)...", flush=True)
    for i, (pid, kernel, ref_src) in enumerate(zip(problem_ids, kernels, ref_sources)):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:3d}/{len(kernels)}] Problem {pid:3d}...", end=" ", flush=True)

        result_path = model_dir / f"problem_{pid:03d}_result.json"

        # Skip if already evaluated
        if result_path.exists():
            with open(result_path) as f:
                r = json.load(f)
            r["problem_id"] = pid
            results.append(r)
            if (i + 1) % 10 == 0 or i == 0:
                status = "CORRECT" if r.get("correct") else ("COMPILED" if r.get("compiled") else "FAIL")
                speedup_str = f" {r.get('speedup', 0):.2f}x" if r.get("speedup", 0) > 0 else ""
                print(f"{status}{speedup_str} (cached)", flush=True)
            continue

        if kernel is None:
            r = {
                "problem_id": pid, "compiled": False, "correct": False,
                "speedup": 0.0, "kernel_ms": -1, "ref_ms": -1,
                "error": "no code extracted from model output",
                "static_check": "skip",
            }
            if (i + 1) % 10 == 0 or i == 0:
                print("SKIP (no code)", flush=True)
            results.append(r)
            with open(result_path, "w") as f:
                json.dump(r, f, indent=2, default=str)
            continue

        # Static checker (runs in main process - safe)
        try:
            static_ok, static_err, static_warn = validate_kernel_static(
                kernel, backend=BACKEND, precision=PRECISION
            )
        except Exception:
            static_ok, static_err, static_warn = True, "", []

        # Save kernel and ref for subprocess
        kernel_file = model_dir / f"problem_{pid:03d}_kernel.py"
        ref_file = model_dir / f"problem_{pid:03d}_ref.py"
        kernel_file.write_text(kernel)
        ref_file.write_text(ref_src)

        # Run evaluation in subprocess
        try:
            proc = subprocess.run(
                [sys.executable, "/data/kb_eval_worker.py",
                 str(kernel_file), str(ref_file), str(result_path)],
                capture_output=True, text=True, timeout=300,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0",
                     "TORCH_EXTENSIONS_DIR": "/data/.torch_extensions",
                     "TMPDIR": "/data/tmp"},
            )
            if result_path.exists():
                with open(result_path) as f:
                    r = json.load(f)
            else:
                stderr_tail = (proc.stderr or "")[-300:]
                r = {"compiled": False, "correct": False, "speedup": 0.0,
                     "kernel_ms": -1, "ref_ms": -1,
                     "error": f"worker failed (rc={proc.returncode}): {stderr_tail}"}
        except subprocess.TimeoutExpired:
            r = {"compiled": False, "correct": False, "speedup": 0.0,
                 "kernel_ms": -1, "ref_ms": -1, "error": "evaluation timeout (300s)"}
        except Exception as e:
            r = {"compiled": False, "correct": False, "speedup": 0.0,
                 "kernel_ms": -1, "ref_ms": -1, "error": f"subprocess error: {str(e)[:300]}"}

        r["problem_id"] = pid
        r["static_check"] = "pass" if static_ok else f"fail: {static_err}"
        if static_warn:
            r["static_warnings"] = static_warn

        status = "CORRECT" if r.get("correct") else ("COMPILED" if r.get("compiled") else "FAIL")
        speedup_str = f" {r.get('speedup', 0):.2f}x" if r.get("speedup", 0) > 0 else ""
        if (i + 1) % 10 == 0 or i == 0:
            print(f"{status}{speedup_str}", flush=True)

        results.append(r)
        with open(result_path, "w") as f:
            json.dump(r, f, indent=2, default=str)

    # Aggregate
    n_total = len(results)
    n_extracted = sum(1 for k in kernels if k is not None)
    n_compiled = sum(1 for r in results if r["compiled"])
    n_correct = sum(1 for r in results if r["correct"])
    speedups = [r["speedup"] for r in results if r["correct"] and r["speedup"] > 0]
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0
    median_speedup = sorted(speedups)[len(speedups)//2] if speedups else 0.0
    n_faster = sum(1 for s in speedups if s > 1.0)
    n_static_fail = sum(1 for r in results if "fail" in str(r.get("static_check", "")))

    # KernelBench composite score
    compile_rate = n_compiled / n_total
    correct_rate = n_correct / n_total
    speedup_score = min(avg_speedup / 2.0, 1.0)
    faster_rate = n_faster / n_total
    composite = 0.10 * compile_rate + 0.40 * correct_rate + 0.30 * speedup_score + 0.20 * faster_rate

    summary = {
        "model": model_name,
        "level": LEVEL,
        "total_problems": n_total,
        "code_extracted": n_extracted,
        "compiled": n_compiled,
        "compile_rate": compile_rate,
        "correct": n_correct,
        "correct_rate": correct_rate,
        "avg_speedup": avg_speedup,
        "median_speedup": median_speedup,
        "faster_than_ref": n_faster,
        "faster_rate": faster_rate,
        "static_check_fails": n_static_fail,
        "generation_time_s": gen_time,
        "total_gen_tokens": total_gen_tokens,
        "composite_score": composite,
        "all_speedups": speedups,
    }

    with open(model_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(model_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'─'*50}")
    print(f"  {short_name} RESULTS")
    print(f"{'─'*50}")
    print(f"  Code extracted: {n_extracted}/{n_total}")
    print(f"  Compiled:       {n_compiled}/{n_total} ({100*compile_rate:.1f}%)")
    print(f"  Correct:        {n_correct}/{n_total} ({100*correct_rate:.1f}%)")
    print(f"  Avg Speedup:    {avg_speedup:.3f}x (correct only)")
    print(f"  Median Speedup: {median_speedup:.3f}x")
    print(f"  Faster than ref:{n_faster}/{n_total}")
    print(f"  Static fails:   {n_static_fail}")
    print(f"  COMPOSITE:      {composite:.4f}")
    print(f"{'─'*50}", flush=True)

    return summary


def main():
    os.environ["TORCH_EXTENSIONS_DIR"] = "/data/.torch_extensions"
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
    os.environ["TMPDIR"] = "/data/tmp"
    os.environ["XDG_CACHE_HOME"] = "/data/cache"
    os.makedirs("/data/tmp", exist_ok=True)
    os.makedirs(HF_CACHE, exist_ok=True)
    os.makedirs("/data/.torch_extensions", exist_ok=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    print("="*70)
    print("KernelBench Evaluation — Laguna XS2 Dense K8 CUDA Models")
    print("="*70)
    print(f"Level: {LEVEL} | Problems: {NUM_PROBLEMS}")
    print(f"Correctness trials: {NUM_CORRECT_TRIALS} | Perf trials: {NUM_PERF_TRIALS}")
    print(f"GPU arch: {GPU_ARCH} | Precision: {PRECISION}")
    print(f"Max new tokens: {MAX_NEW_TOKENS} | Temperature: {TEMPERATURE}")
    print("="*70, flush=True)

    print("\nLoading KernelBench dataset...", flush=True)
    dataset = construct_kernelbench_dataset(level=LEVEL, source="huggingface")
    print(f"Loaded {len(dataset.get_problem_ids())} problems", flush=True)

    all_summaries = []
    for model_name in MODELS:
        try:
            summary = run_eval_for_model(model_name, dataset, device)
            all_summaries.append(summary)
        except Exception as e:
            print(f"\nERROR evaluating {model_name}: {e}", flush=True)
            traceback.print_exc()
            all_summaries.append({"model": model_name, "error": str(e)})

    # Final comparison
    print("\n" + "="*90)
    print("FINAL COMPARISON — KernelBench Level 1 (100 problems)")
    print("="*90)
    print(f"{'Model':<35} {'Extract':>8} {'Compile':>8} {'Correct':>8} {'AvgSpd':>7} {'MedSpd':>7} {'Faster':>7} {'Score':>7}")
    print("-"*90)
    for s in all_summaries:
        if "error" in s:
            print(f"{s['model'].split('/')[-1]:<35} ERROR: {s['error'][:50]}")
        else:
            print(f"{s['model'].split('/')[-1]:<35} "
                  f"{s['code_extracted']:>7d} "
                  f"{100*s['compile_rate']:>7.1f}% "
                  f"{100*s['correct_rate']:>7.1f}% "
                  f"{s['avg_speedup']:>6.3f}x "
                  f"{s['median_speedup']:>6.3f}x "
                  f"{s['faster_than_ref']:>6d} "
                  f"{s['composite_score']:>6.4f}")

    print("\nComposite = 10%*compile + 40%*correct + 30%*speedup_norm + 20%*faster_rate")
    print(f"\nResults saved to {OUTPUT_DIR}")

    with open(OUTPUT_DIR / "comparison.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)


if __name__ == "__main__":
    main()
