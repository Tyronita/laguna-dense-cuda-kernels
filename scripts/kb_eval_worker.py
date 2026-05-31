"""Worker script: evaluates a single kernel in isolation. Called as subprocess."""
import json
import os
import sys
import torch

sys.path.insert(0, "/data/shinka-kernelbench/KernelBench/src")
from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string
from kernelbench.timing import measure_ref_program_time
from kernelbench.utils import set_gpu_arch

def main():
    kernel_path = sys.argv[1]
    ref_path = sys.argv[2]
    output_path = sys.argv[3]

    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/data/.torch_extensions")
    os.environ.setdefault("TMPDIR", "/data/tmp")

    with open(kernel_path) as f:
        kernel_src = f.read()
    with open(ref_path) as f:
        ref_src = f.read()

    set_gpu_arch(["Ampere"])
    device = torch.device("cuda:0")

    try:
        eval_result = eval_kernel_against_ref(
            original_model_src=ref_src,
            custom_model_src=kernel_src,
            measure_performance=True,
            timing_method="cuda_event",
            verbose=False,
            num_correct_trials=5,
            num_perf_trials=100,
            device=device,
            backend="cuda",
            precision=get_torch_dtype_from_string("fp32"),
        )
    except Exception as e:
        result = {"compiled": False, "correct": False, "speedup": 0.0,
                  "kernel_ms": -1, "ref_ms": -1, "error": str(e)[:500]}
        with open(output_path, "w") as f:
            json.dump(result, f)
        return

    if eval_result is None:
        result = {"compiled": False, "correct": False, "speedup": 0.0,
                  "kernel_ms": -1, "ref_ms": -1, "error": "eval returned None"}
        with open(output_path, "w") as f:
            json.dump(result, f)
        return

    compiled = eval_result.compiled
    correct = eval_result.correctness
    kern_ms = eval_result.runtime if eval_result.runtime and eval_result.runtime > 0 else -1
    error = ""
    if not compiled:
        error = str(eval_result.metadata.get("compilation_error", "compile fail"))[:500]
    elif not correct:
        error = "correctness check failed"

    ref_ms = -1
    speedup = 0.0
    if correct and kern_ms > 0:
        try:
            ref_time_result = measure_ref_program_time(
                ref_arch_name="ref", ref_arch_src=ref_src,
                num_trials=100, use_torch_compile=False,
                timing_method="cuda_event", device=device,
                verbose=False, precision="fp32",
            )
            if ref_time_result:
                ref_ms = ref_time_result["mean"]
                speedup = ref_ms / kern_ms
        except Exception:
            pass

    result = {"compiled": compiled, "correct": correct, "speedup": speedup,
              "kernel_ms": kern_ms, "ref_ms": ref_ms, "error": error}
    with open(output_path, "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
