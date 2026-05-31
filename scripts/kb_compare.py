import json, glob, os, sys

base = "/data/kernelbench-laguna-eval"
models = [
    ("GRPO-offline", "laguna-xs2-dense-k8-cuda-grpo"),
    ("GRPO-online",  "laguna-xs2-dense-k8-cuda-rft"),
    ("DPO",          "laguna-xs2-dense-k8-cuda-dpo"),
    ("SFT-v1",       "laguna-xs2-dense-k8-cuda-sft"),
    ("SFT-v2",       "laguna-xs2-dense-k8-cuda-sft-v2"),
]

all_results = {}
for label, dirname in models:
    path = os.path.join(base, dirname)
    results = []
    for f in sorted(glob.glob(os.path.join(path, "problem_*_result.json"))):
        results.append(json.load(open(f)))
    all_results[label] = results

print("=" * 90)
print("KernelBench L1 — All Models Comparison")
print("=" * 90)
print("")
print("%-15s %5s %8s %8s %8s %10s %10s %7s" % (
    "Model", "Eval", "Compile", "Correct", "Faster", "Avg Spdup", "Med Spdup", "fast_0"))
print("-" * 90)

for label, dirname in models:
    results = all_results[label]
    n = len(results)
    if n == 0:
        print("%-15s   0/100 (no results yet)" % label)
        continue
    compiled = sum(1 for r in results if r.get("compiled"))
    correct = sum(1 for r in results if r.get("correct"))
    speedups = [r["speedup"] for r in results if r.get("correct") and r.get("speedup", 0) > 0]
    faster = sum(1 for s in speedups if s > 1.0)
    avg_sp = sum(speedups)/len(speedups) if speedups else 0
    med_sp = sorted(speedups)[len(speedups)//2] if speedups else 0
    print("%-15s %3d%s %7d%% %7d%% %8d %9.3fx %9.3fx %5d%%" % (
        label,
        n, "" if n == 100 else "*",
        int(100.0*compiled/n),
        int(100.0*correct/n),
        faster,
        avg_sp, med_sp,
        int(100.0*correct/n)))

# Per-problem comparison for completed models
done_models = [(l, d) for l, d in models if len(all_results[l]) == 100]
if len(done_models) >= 2:
    print("")
    print("=" * 90)
    print("Per-Problem Comparison (completed models only)")
    print("=" * 90)

    header = "%-4s " % "PID"
    for label, _ in done_models:
        header += "%-15s " % label
    print(header)
    print("-" * 90)

    for pid in range(1, 101):
        row = "%3d  " % pid
        for label, dirname in done_models:
            results = all_results[label]
            r = next((x for x in results if x.get("problem_id") == pid), None)
            if r is None:
                row += "%-15s " % "?"
            elif r.get("correct"):
                sp = r.get("speedup", 0)
                row += "%-15s " % ("OK %.3fx" % sp)
            elif r.get("compiled"):
                row += "%-15s " % "COMPILED"
            else:
                err = r.get("error", "")[:10]
                row += "%-15s " % ("FAIL %s" % err)
        print(row)

    # Which problems did one model get right that the other didn't?
    print("")
    print("=" * 90)
    print("Differential Analysis")
    print("=" * 90)
    for i, (l1, d1) in enumerate(done_models):
        for l2, d2 in done_models[i+1:]:
            r1 = all_results[l1]
            r2 = all_results[l2]
            r1_correct = set(r.get("problem_id") for r in r1 if r.get("correct"))
            r2_correct = set(r.get("problem_id") for r in r2 if r.get("correct"))
            only1 = r1_correct - r2_correct
            only2 = r2_correct - r1_correct
            both = r1_correct & r2_correct
            print("")
            print("%s vs %s:" % (l1, l2))
            print("  Both correct: %d  %s" % (len(both), sorted(both) if both else ""))
            print("  Only %s: %d  %s" % (l1, len(only1), sorted(only1) if only1 else ""))
            print("  Only %s: %d  %s" % (l2, len(only2), sorted(only2) if only2 else ""))
