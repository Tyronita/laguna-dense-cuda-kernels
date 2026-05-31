import json, glob, os, sys
from collections import defaultdict

base = "/data/kernelbench-laguna-eval"
models = [
    ("GRPO-offline", "laguna-xs2-dense-k8-cuda-grpo"),
    ("GRPO-online",  "laguna-xs2-dense-k8-cuda-rft"),
    ("DPO",          "laguna-xs2-dense-k8-cuda-dpo"),
    ("SFT-v1",       "laguna-xs2-dense-k8-cuda-sft"),
    ("SFT-v2",       "laguna-xs2-dense-k8-cuda-sft-v2"),
]

def categorize_error(r):
    err = r.get("error", "")
    if not err:
        if r.get("correct"):
            return "correct"
        elif r.get("compiled"):
            return "compiled_but_incorrect"
        return "unknown_success"
    if "no code" in err:
        return "no_code_extracted"
    if "__init__() takes" in err:
        return "init_missing_args"
    if "Error building" in err or "Ninja" in err:
        return "cuda_build_error"
    if "correctness" in err:
        return "compiled_but_incorrect"
    if "illegal memory" in err:
        return "cuda_illegal_memory"
    if "Syntax error" in err:
        return "python_syntax_error"
    if "eval returned None" in err:
        return "eval_returned_none"
    if "timeout" in err:
        return "timeout"
    if "worker failed" in err:
        return "worker_crash"
    return "other"

# Per-model error breakdown
print("=" * 90)
print("ERROR ANALYSIS — All Models")
print("=" * 90)

all_categories = set()
model_errors = {}
for label, dirname in models:
    path = os.path.join(base, dirname)
    results = []
    for f in sorted(glob.glob(os.path.join(path, "problem_*_result.json"))):
        r = json.load(open(f))
        r["_pid"] = int(os.path.basename(f).split("_")[1])
        results.append(r)

    cats = defaultdict(list)
    for r in results:
        cat = categorize_error(r)
        cats[cat].append(r)
        all_categories.add(cat)
    model_errors[label] = cats

# Summary table
cats_ordered = ["correct", "compiled_but_incorrect", "no_code_extracted",
                "init_missing_args", "cuda_build_error", "cuda_illegal_memory",
                "python_syntax_error", "eval_returned_none", "worker_crash", "timeout", "other"]

print("")
print("%-25s" % "Error Category", end="")
for label, _ in models:
    print(" %12s" % label, end="")
print("")
print("-" * 90)
for cat in cats_ordered:
    if cat not in all_categories:
        continue
    print("%-25s" % cat, end="")
    for label, _ in models:
        count = len(model_errors[label].get(cat, []))
        print(" %12d" % count, end="")
    print("")

# Detailed CUDA build errors
print("")
print("=" * 90)
print("CUDA BUILD ERRORS — Detailed")
print("=" * 90)
for label, dirname in models:
    errs = model_errors[label].get("cuda_build_error", [])
    if not errs:
        continue
    print("")
    print("--- %s (%d build errors) ---" % (label, len(errs)))
    # Group by error message pattern
    err_patterns = defaultdict(list)
    for r in errs:
        err = r.get("error", "")
        # Extract the key part of the build error
        if "error:" in err:
            # Try to get the compiler error line
            for line in err.split("\\n"):
                if "error:" in line.lower():
                    err = line.strip()[:120]
                    break
        err_patterns[err[:100]].append(r["_pid"])
    for pattern, pids in sorted(err_patterns.items(), key=lambda x: -len(x[1])):
        print("  [%d] P%s" % (len(pids), ",".join(str(p) for p in pids[:10])))
        print("       %s" % pattern)

# Detailed CUDA illegal memory
print("")
print("=" * 90)
print("CUDA ILLEGAL MEMORY ACCESS — Detailed")
print("=" * 90)
for label, dirname in models:
    errs = model_errors[label].get("cuda_illegal_memory", [])
    if not errs:
        continue
    print("")
    print("--- %s (%d illegal memory errors) ---" % (label, len(errs)))
    for r in errs:
        pid = r["_pid"]
        # Read the kernel to see what went wrong
        kernel_path = os.path.join(base, dirname, "kernels", "problem_%03d_kernel.py" % pid)
        if not os.path.exists(kernel_path):
            kernel_path = os.path.join(base, dirname, "problem_%03d_raw.txt" % pid)
        kernel = open(kernel_path).read() if os.path.exists(kernel_path) else "(no file)"

        # Find the kernel function
        kernel_lines = kernel.split("\n")
        global_lines = [l for l in kernel_lines if "__global__" in l or "blockIdx" in l or "threadIdx" in l]

        print("  P%d:" % pid)
        for gl in global_lines[:5]:
            print("    %s" % gl.strip())
        if not global_lines:
            print("    (no kernel function found)")

# Compiled but incorrect — what went wrong?
print("")
print("=" * 90)
print("COMPILED BUT INCORRECT — Per Problem")
print("=" * 90)
for label, dirname in models:
    errs = model_errors[label].get("compiled_but_incorrect", [])
    if not errs:
        continue
    print("")
    print("--- %s (%d compiled but incorrect) ---" % (label, len(errs)))
    for r in errs:
        pid = r["_pid"]
        print("  P%3d  %s" % (pid, r.get("error", "")[:80]))

# __init__ missing args
print("")
print("=" * 90)
print("__init__ MISSING ARGS — Which Problems")
print("=" * 90)
all_init_pids = set()
for label, dirname in models:
    errs = model_errors[label].get("init_missing_args", [])
    for r in errs:
        all_init_pids.add(r["_pid"])
print("Problems affected: %s" % sorted(all_init_pids))
print("Total unique: %d" % len(all_init_pids))
print("These are all ops requiring constructor args (conv weights, norm params, etc)")
