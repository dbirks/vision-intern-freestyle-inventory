#!/usr/bin/env python3
"""GPU smoke test for k8s jobs: prove the card is visible, usable, and fast enough.

Run by k8s/submit.sh smoke (see k8s/README.md). Contract:
  * env OUT_DIR (default /out)  -> writes <OUT_DIR>/smoke.json
  * env VI_GPU_MEM_MB           -> the VRAM this job is allowed to touch. In `hami` mode that
    is already a hard cap enforced by HAMi's libvgpu; in `share` mode (no HAMi booking, raw
    card next to a resident LLM engine) we cap torch's allocator to it ourselves so a smoke
    test can never OOM the serving pod that pays our wages.
  * env VI_EXPECT_GPU           -> optional substring the device name must contain; a
    mismatch fails the job (that is how we prove the card pin actually worked).
  * env VI_MATMUL_N             -> matrix side, default 4096.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def sh(cmd: list[str]) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (p.stdout or p.stderr).strip()
    except Exception as exc:  # noqa: BLE001 - diagnostics only, never fatal
        return f"<unavailable: {exc}>"


def nvidia_smi() -> dict:
    """Real driver view (physical card). In hami mode this is the patched/quota view."""
    out = sh(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in out.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 6 and f[2].replace(".", "").isdigit():
            gpus.append(
                {
                    "name": f[0],
                    "driver": f[1],
                    "memory_total_mib": int(float(f[2])),
                    "memory_used_mib": int(float(f[3])),
                    "memory_free_mib": int(float(f[4])),
                    "power_w": f[5],
                }
            )
    return {"raw": out, "gpus": gpus}


def time_matmul(n: int, dtype_name: str, inplace: bool, seconds_budget: float = 4.0) -> dict:
    """4096x4096 matmul, timed. inplace=True writes the result over `a` (2 live matrices
    instead of 3) -- what we fall back to when the card is 98 % full of somebody else's KV
    cache and every extra MiB matters."""
    import torch

    dtype = getattr(torch, dtype_name)
    dev = torch.device("cuda")
    a = torch.rand(n, n, device=dev, dtype=dtype)
    b = torch.rand(n, n, device=dev, dtype=dtype)
    c = a if inplace else torch.empty_like(a)
    for _ in range(3):  # warm up cublas handles / kernels
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    iters, elapsed = 0, 0.0
    while elapsed < seconds_budget:
        t0 = time.perf_counter()
        for _ in range(5):
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize()
        elapsed += time.perf_counter() - t0
        iters += 5
    flops = 2.0 * n**3 * iters / elapsed
    del a, b, c
    torch.cuda.empty_cache()
    return {
        "iters": iters,
        "seconds": round(elapsed, 3),
        "tflops": round(flops / 1e12, 2),
        "in_place": inplace,
        "bytes_of_tensors": (2 if inplace else 3) * n * n * torch.tensor([], dtype=dtype).element_size(),
    }


def main() -> int:
    out_dir = os.environ.get("OUT_DIR", "/out")
    mode = os.environ.get("VI_GPU_MODE", "hami")
    budget_mib = env_int("VI_GPU_MEM_MB", 4096)
    n = env_int("VI_MATMUL_N", 4096)
    expect = os.environ.get("VI_EXPECT_GPU", "")

    report: dict = {
        "job": os.environ.get("VI_JOB_NAME", ""),
        "node": socket.gethostname(),
        "gpu_mode": mode,
        "vram_budget_mib": budget_mib,
        "matmul_n": n,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rc = 0
    os.makedirs(out_dir, exist_ok=True)

    def finish() -> int:
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        path = os.path.join(out_dir, "smoke.json")
        with open(path, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        print(f"wrote {path}", file=sys.stderr)
        return rc

    smi = nvidia_smi()
    report["nvidia_smi"] = smi
    phys_free_mib = smi["gpus"][0]["memory_free_mib"] if smi["gpus"] else 0
    print("== nvidia-smi:", smi["raw"], flush=True)

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"torch import failed: {exc}"
        print(report["error"], file=sys.stderr)
        return finish()

    report["torch"] = {
        "version": torch.__version__,
        "cuda_builtin": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
    }
    print(f"== torch {torch.__version__} (cuda {torch.version.cuda})", flush=True)
    if not report["torch"]["cuda_available"]:
        report["error"] = "torch.cuda.is_available() is False"
        return finish()

    dev = torch.device("cuda")
    free_b, total_b = torch.cuda.mem_get_info(dev)
    props = torch.cuda.get_device_properties(dev)
    report["device"] = {
        "name": props.name,
        "capability": f"sm_{props.major}{props.minor}",
        "total_memory_mib": round(total_b / 2**20),
        "free_memory_mib_at_start": round(free_b / 2**20),
        "multi_processor_count": props.multi_processor_count,
    }
    print(
        f"== {props.name} sm_{props.major}{props.minor}"
        f"  {round(total_b / 2**20)} MiB total / {round(free_b / 2**20)} MiB free",
        flush=True,
    )

    # In share mode we are a guest on a card that a serving engine already lives on: never
    # take more than our declared budget, and never more than what is physically free.
    # `free_b` is measured after the CUDA context exists, so what is left is what the
    # allocator may really use. Keep a margin for cuBLAS workspaces: on a card that is 97 %
    # full, overshooting is the difference between failing politely and taking a resident
    # LLM engine down with us.
    reserve = env_int("VI_RESERVE_MIB", 96)
    cap = round(free_b / 2**20) - reserve
    if mode == "share" and budget_mib > 0:
        cap = min(cap, budget_mib)
    cap = max(0, cap)
    report["allocator_cap_mib"] = cap
    if cap and cap * 2**20 < total_b:
        torch.cuda.set_per_process_memory_fraction(cap / (total_b / 2**20), 0)
        print(f"== allocator capped to {cap} MiB of {round(total_b / 2**20)} MiB", flush=True)
    if mode == "share":
        report["physical_free_mib_at_start"] = phys_free_mib

    if expect and expect.lower() not in props.name.lower():
        report["error"] = f"expected a GPU matching {expect!r}, got {props.name!r}"
        rc = 1
        return finish()

    try:
        # dtype list: float32 is 4x the bytes of fp16, so only attempt it when the cap allows
        # (a 4096^2 fp32 matmul wants 3 x 64 MiB of tensors on top of the CUDA context).
        # Pick a dtype that fits what the allocator may actually use. A 4096^2 matmul is
        # 32 MiB per fp16 matrix, and on this cluster the card is usually ~98 % full of a
        # resident LLM engine, so adapt instead of OOMing: in-place first, then narrower
        # dtypes, and only widen when there is room for the 3-matrix (non-in-place) form.
        cap_mib = report.get("allocator_cap_mib") or round(free_b / 2**20)
        mib = 2**20
        # (dtype, in_place, tensors_at_n=4096_MiB, n_when_that_does_not_fit)
        ladder = [
            ("float16", False, 96, n), ("bfloat16", False, 96, n), ("float32", False, 192, n),
            ("float16", True, 64, n), ("bfloat16", True, 64, n),
            ("float16", True, 16, max(1024, n // 2)),
        ]
        picks = [x for x in ladder if x[2] + 48 <= cap_mib] or [ladder[3]]
        report["matmul"] = {}
        notes = []
        for dt, inplace, need, nn in picks:
            tag = f"{dt}{'' if nn == n else f'@{nn}'}{'-inplace' if inplace else ''}"
            try:
                t = time_matmul(nn, dt, inplace)
                report["matmul"][tag] = t
                print(f"== {nn}x{nn} {tag}: {t['tflops']} TFLOPS "
                      f"({t['iters']} iters, {need} MiB of tensors)", flush=True)
            except Exception as exc:  # noqa: BLE001 - try the next rung of the ladder
                notes.append(f"{tag}: {type(exc).__name__}")
                print(f"== {tag} failed ({type(exc).__name__}), "
                      f"free: {round(torch.cuda.mem_get_info(dev)[0] / mib)} MiB", flush=True)
                torch.cuda.empty_cache()
        if not report["matmul"]:
            report["error"] = "no matmul configuration fit in " + str(cap_mib) + " MiB: " + "; ".join(notes)
            rc = 1
        elif notes:
            report["matmul_notes"] = notes
        report["peak_memory_mib"] = round(torch.cuda.max_memory_allocated(dev) / 2**20)
        report["peak_memory_reserved_mib"] = round(torch.cuda.max_memory_reserved(dev) / 2**20)
        report["nvidia_smi_after"] = nvidia_smi()
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"matmul failed: {type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
        rc = 1
    return finish()


if __name__ == "__main__":
    sys.exit(main())
