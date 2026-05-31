import json, glob, os, sys
sys.path.insert(0, "/data/shinka-kernelbench/KernelBench/src")
from kernelbench.dataset import construct_kernelbench_dataset
ds = construct_kernelbench_dataset(level=1, source="huggingface")

results = {}
for f in sorted(glob.glob("problem_*_result.json")):
    pid = int(os.path.basename(f).split("_")[1])
    results[pid] = json.load(open(f))

categories = {
    "matmul (1-18)": list(range(1, 19)),
    "activation (19-32)": list(range(19, 33)),
    "norm (33-40)": list(range(33, 41)),
    "pooling (41-49)": list(range(41, 50)),
    "conv (50-87)": list(range(50, 88)),
    "other (88-100)": list(range(88, 101)),
}

print("=" * 70)
print("GRPO-OFFLINE (cuda-grpo) -- KernelBench L1 Full Analysis")
print("=" * 70)

n = len(results)
no_code = sum(1 for r in results.values() if "no code" in r.get("error", ""))
compiled = sum(1 for r in results.values() if r.get("compiled"))
correct = sum(1 for r in results.values() if r.get("correct"))
speedups = [r["speedup"] for r in results.values() if r.get("correct") and r.get("speedup", 0) > 0]
faster = sum(1 for s in speedups if s > 1.0)

print("")
print("OVERALL")
print("-" * 40)
print("  Total problems:    %d" % n)
print("  No code extracted: %d" % no_code)
print("  Compiled:          %d/%d (%.1f%%)" % (compiled, n, 100.0*compiled/n))
print("  Correct:           %d/%d (%.1f%%)" % (correct, n, 100.0*correct/n))
print("  Faster than ref:   %d/%d" % (faster, n))
if speedups:
    print("  Avg speedup:       %.3fx" % (sum(speedups)/len(speedups)))
    print("  Median speedup:    %.3fx" % sorted(speedups)[len(speedups)//2])

print("")
print("BY CATEGORY")
print("-" * 70)
print("%-20s %6s %8s %8s %8s %12s" % ("Category", "Total", "Compile", "Correct", "Faster", "Avg Speedup"))
print("-" * 70)
for cat, pids in categories.items():
    cat_r = [results[p] for p in pids if p in results]
    cat_n = len(cat_r)
    cat_comp = sum(1 for r in cat_r if r.get("compiled"))
    cat_corr = sum(1 for r in cat_r if r.get("correct"))
    cat_sp = [r["speedup"] for r in cat_r if r.get("correct") and r.get("speedup", 0) > 0]
    cat_fast = sum(1 for s in cat_sp if s > 1.0)
    avg_sp = "%.3fx" % (sum(cat_sp)/len(cat_sp)) if cat_sp else "-"
    print("%-20s %6d %8d %8d %8d %12s" % (cat, cat_n, cat_comp, cat_corr, cat_fast, avg_sp))

print("")
print("ERROR BREAKDOWN")
print("-" * 70)
err_cats = {}
for r in results.values():
    err = r.get("error", "")
    if not err:
        key = "success (correct)"
    elif "no code" in err:
        key = "no code extracted"
    elif "__init__() takes" in err:
        key = "__init__ missing args (wrapper)"
    elif "Error building" in err:
        key = "CUDA compilation error"
    elif "correctness" in err:
        key = "compiled but incorrect"
    elif "illegal memory" in err:
        key = "CUDA illegal memory access"
    elif "Syntax error" in err:
        key = "Python syntax error"
    elif "eval returned None" in err:
        key = "eval returned None"
    elif "timeout" in err:
        key = "timeout"
    else:
        key = "other: " + err[:40]
    err_cats[key] = err_cats.get(key, 0) + 1

for k, v in sorted(err_cats.items(), key=lambda x: -x[1]):
    print("  %3d  %s" % (v, k))

print("")
print("CORRECT KERNELS (detail)")
print("-" * 70)
print("%4s  %-40s %8s %9s %9s" % ("PID", "Problem", "Speedup", "Kern_ms", "Ref_ms"))
print("-" * 70)
for pid in sorted(results.keys()):
    r = results[pid]
    if r.get("correct"):
        name = ds.get_problem_by_id(pid).name[:40]
        print("%4d  %-40s %7.3fx %8.2fms %8.2fms" % (
            pid, name, r.get("speedup", 0), r.get("kernel_ms", -1), r.get("ref_ms", -1)))

print("")
print("COMPILED BUT INCORRECT (could be close)")
print("-" * 70)
for pid in sorted(results.keys()):
    r = results[pid]
    if r.get("compiled") and not r.get("correct"):
        name = ds.get_problem_by_id(pid).name[:40]
        print("%4d  %-40s  %s" % (pid, name, r.get("error", "")[:40]))
