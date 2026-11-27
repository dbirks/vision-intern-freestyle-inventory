#!/usr/bin/env python3
"""Fault-tolerant status collector for the Freestyle dev log.

Collects worker status, Kubernetes jobs/pods, GPU memory, git history,
cached unit test counts, and experiment metrics into ``devlog/site/status.json``.
Every collector step catches errors and records an ``unavailable`` note rather
than crashing the run.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = Path(__file__).resolve().parent / "site"
STATUS_PATH = SITE_DIR / "status.json"
PYTEST_CACHE_PATH = SITE_DIR / ".pytest_cache.json"

PI_ORCH_SCRIPT = Path.home() / ".claude/skills/pi-orch/scripts/pi-orch"
EXT_LOGS_DIR = Path.home() / ".cache/pi-orch/vision-intern/ext"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _format_duration(seconds: float | int | None) -> str:
    if seconds is None or seconds < 0:
        return "-"
    secs = int(seconds)
    if secs < 60:
        return f"{secs}s"
    mins, secs = divmod(secs, 60)
    if mins < 60:
        return f"{mins}m{secs}s"
    hours, mins = divmod(mins, 60)
    if hours < 24:
        return f"{hours}h{mins}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours}h"


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _run(
    cmd: list[str],
    *,
    timeout: int = 15,
    cwd: Path = REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _extract_last_report_block(text: str, max_lines: int = 40) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    status_idx = None
    for idx in range(len(lines) - 1, -1, -1):
        if re.match(r"^(?:#+\s*)?(?:\*\*)?STATUS\b", lines[idx].strip(), re.IGNORECASE):
            status_idx = idx
            break
    if status_idx is not None:
        block = lines[status_idx : status_idx + max_lines]
    else:
        non_empty = [line for line in lines if line.strip()]
        block = non_empty[-min(20, len(non_empty)) :]
    return "\n".join(block).strip()


def collect_pi_workers() -> dict[str, Any]:
    """Collect pi-orch worker status table and last report blocks for finished workers."""
    if not PI_ORCH_SCRIPT.exists():
        return {
            "ok": False,
            "status": "unavailable",
            "error": f"pi-orch script not found at {PI_ORCH_SCRIPT}",
            "workers": [],
        }
    proc = _run([str(PI_ORCH_SCRIPT), "status"], timeout=10)
    if proc.returncode != 0:
        return {
            "ok": False,
            "status": "unavailable",
            "error": (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip(),
            "workers": [],
        }

    workers: list[dict[str, Any]] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("SLUG"):
            continue
        parts = line.split(None, 5)
        if len(parts) < 5:
            continue
        slug, state, age, idle, session = parts[:5]
        report_inline = parts[5].strip() if len(parts) > 5 else ""
        report_block = ""
        if state in ("done", "dead") or report_inline:
            rep = _run([str(PI_ORCH_SCRIPT), "report", slug], timeout=10)
            if rep.returncode == 0 and rep.stdout.strip():
                report_block = _extract_last_report_block(rep.stdout)
        workers.append(
            {
                "slug": slug,
                "state": state,
                "age": age,
                "idle": idle,
                "session": session,
                "report_inline": report_inline,
                "report_block": report_block,
            }
        )

    counts = {
        "running": sum(1 for w in workers if w["state"] == "running"),
        "done": sum(1 for w in workers if w["state"] == "done"),
        "dead": sum(1 for w in workers if w["state"] == "dead"),
    }
    return {
        "ok": True,
        "status": "ok",
        "counts": counts,
        "workers": workers,
    }


def _extract_ext_status(lines: list[str]) -> str:
    for idx in range(len(lines) - 1, -1, -1):
        stripped = lines[idx].strip()
        m = re.match(r"^(?:#+\s*)?(?:\*\*)?STATUS(?:\*\*)?\s*:?\s*(.*)$", stripped, re.IGNORECASE)
        if m:
            rest = m.group(1).strip().strip("*").strip()
            if rest:
                return rest
            for nxt in lines[idx + 1 : idx + 4]:
                if nxt.strip():
                    return nxt.strip()
            return stripped
    for line in reversed(lines):
        if line.strip():
            return line.strip()[:220]
    return "(empty log)"


def _read_tail(path: Path, max_bytes: int = 32768, max_lines: int = 120) -> list[str]:
    size = path.stat().st_size
    if size == 0:
        return []
    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
        data = fh.read().decode("utf-8", errors="replace")
    return data.splitlines()[-max_lines:]


def collect_external_agents() -> dict[str, Any]:
    """Collect tail STATUS lines from ~/.cache/pi-orch/vision-intern/ext/*.log."""
    if not EXT_LOGS_DIR.exists():
        return {
            "ok": False,
            "status": "unavailable",
            "error": f"Directory not found: {EXT_LOGS_DIR}",
            "latest_by_agent": [],
            "recent": [],
        }

    logs = sorted(
        EXT_LOGS_DIR.glob("*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    latest_by_agent: dict[str, dict[str, Any]] = {}
    recent: list[dict[str, Any]] = []

    for log_path in logs:
        stat = log_path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        stem = log_path.stem
        agent_name = stem.split("-", 1)[0] if "-" in stem else stem
        lines = _read_tail(log_path)
        status_line = _extract_ext_status(lines) if lines else "(active or empty log)"
        tail_lines = [ln for ln in lines[-6:] if ln.strip()]
        entry = {
            "agent": agent_name,
            "file": log_path.name,
            "stem": stem,
            "size_bytes": stat.st_size,
            "mtime": mtime.isoformat(),
            "status_line": status_line,
            "tail": "\n".join(tail_lines),
        }
        recent.append(entry)
        if agent_name not in latest_by_agent:
            latest_by_agent[agent_name] = entry
        elif latest_by_agent[agent_name]["size_bytes"] == 0 and stat.st_size > 0:
            # Keep track of newest non-empty log if the very latest file is 0 bytes.
            latest_by_agent[agent_name]["latest_nonempty"] = entry

    return {
        "ok": True,
        "status": "ok",
        "total_logs": len(logs),
        "latest_by_agent": list(latest_by_agent.values()),
        "recent": recent[:16],
    }


def collect_k8s() -> dict[str, Any]:
    """Collect k8s jobs and pods labelled app=vision-intern."""
    proc = _run(
        [
            "kubectl",
            "--context",
            "admin@home",
            "get",
            "jobs,pods",
            "-l",
            "app=vision-intern",
            "-o",
            "json",
        ],
        timeout=15,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "status": "unavailable",
            "error": (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip(),
            "jobs": [],
            "pods": [],
        }

    payload = json.loads(proc.stdout)
    now = _now_utc()
    raw_jobs: list[dict[str, Any]] = []
    pods: list[dict[str, Any]] = []
    pod_by_job: dict[str, dict[str, Any]] = {}

    for item in payload.get("items", []):
        kind = item.get("kind", "")
        meta = item.get("metadata", {})
        labels = meta.get("labels", {})
        spec = item.get("spec", {})
        st = item.get("status", {})
        name = meta.get("name", "unknown")
        gpu = labels.get("vision-intern/gpu") or labels.get("gpu") or ""

        if kind == "Job":
            raw_jobs.append(item)
        elif kind == "Pod":
            node_name = spec.get("nodeName", "-")
            if not gpu:
                if "8me-tcy" in node_name:
                    gpu = "5090"
                elif "210-73x" in node_name:
                    gpu = "6000"
            phase = st.get("phase", "Unknown")
            # Surface container waiting reason (e.g., ImagePullBackOff, CrashLoopBackOff)
            for cs in st.get("containerStatuses", []) or []:
                waiting = (cs.get("state") or {}).get("waiting")
                if waiting and waiting.get("reason"):
                    phase = f"{phase} ({waiting['reason']})"
                    break
            start = _parse_iso(st.get("startTime"))
            finished_str = None
            term_time = None
            for cs in st.get("containerStatuses", []) or []:
                finished = ((cs.get("state") or {}).get("terminated") or {}).get("finishedAt")
                if finished:
                    finished_str = finished
                    term_time = _parse_iso(finished)
                    break
            end = term_time or now
            dur = _format_duration((end - start).total_seconds()) if start else "-"
            pod_entry = {
                "name": name,
                "gpu": gpu or "-",
                "phase": phase,
                "duration": dur,
                "node": node_name,
                "start_time": st.get("startTime"),
                "finished_at": finished_str,
            }
            pods.append(pod_entry)

            job_owner = labels.get("job-name")
            if not job_owner:
                for ref in meta.get("ownerReferences", []) or []:
                    if ref.get("kind") == "Job" and ref.get("name"):
                        job_owner = ref["name"]
                        break
            if job_owner:
                pod_by_job[job_owner] = pod_entry

    jobs: list[dict[str, Any]] = []
    for item in raw_jobs:
        meta = item.get("metadata", {})
        labels = meta.get("labels", {})
        st = item.get("status", {})
        name = meta.get("name", "unknown")
        pod_info = pod_by_job.get(name) or {}
        node_name = pod_info.get("node") or "-"
        gpu = labels.get("vision-intern/gpu") or labels.get("gpu") or ""
        if not gpu or gpu == "-":
            if "8me-tcy" in node_name:
                gpu = "5090"
            elif "210-73x" in node_name:
                gpu = "6000"
            elif pod_info.get("gpu") and pod_info["gpu"] != "-":
                gpu = pod_info["gpu"]

        if st.get("succeeded"):
            phase = "Complete"
        elif st.get("failed"):
            phase = "Failed"
        elif st.get("active"):
            phase = "Running"
        else:
            phase = "Pending"

        start_str = st.get("startTime")
        comp_str = st.get("completionTime")
        end_str = comp_str
        if not end_str and phase in ("Failed", "Complete"):
            conds = st.get("conditions") or []
            for target_type in (phase, "FailureTarget"):
                for cond in conds:
                    if cond.get("type") == target_type and cond.get("status") == "True":
                        end_str = cond.get("lastTransitionTime")
                        if end_str:
                            break
                if end_str:
                    break
            if not end_str:
                end_str = pod_info.get("finished_at")

        start = _parse_iso(start_str)
        end_dt = _parse_iso(end_str) or now
        dur = _format_duration((end_dt - start).total_seconds()) if start else "-"
        jobs.append(
            {
                "name": name,
                "gpu": gpu or "-",
                "node": node_name,
                "phase": phase,
                "duration": dur,
                "start_time": start_str,
                "completion_time": comp_str,
                "end_time": end_str,
            }
        )

    return {
        "ok": True,
        "status": "ok",
        "jobs": jobs,
        "pods": pods,
    }


def collect_gpus() -> dict[str, Any]:
    """Collect GPU memory per card via cluster gpu-vram-exporter pods (or local nvidia-smi)."""
    cards: list[dict[str, Any]] = []
    proc = _run(
        [
            "kubectl",
            "--context",
            "admin@home",
            "get",
            "pods",
            "-l",
            "app=gpu-vram-exporter",
            "-o",
            "json",
        ],
        timeout=10,
    )
    if proc.returncode == 0:
        try:
            data = json.loads(proc.stdout)
            for pod in data.get("items", []):
                pod_name = pod.get("metadata", {}).get("name")
                node_name = pod.get("spec", {}).get("nodeName", "-")
                phase = pod.get("status", {}).get("phase")
                if not pod_name or phase != "Running":
                    continue
                smi = _run(
                    [
                        "kubectl",
                        "--context",
                        "admin@home",
                        "exec",
                        pod_name,
                        "--",
                        "nvidia-smi",
                        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=5,
                )
                if smi.returncode == 0 and smi.stdout.strip():
                    for row in smi.stdout.strip().splitlines():
                        parts = [p.strip() for p in row.split(",")]
                        if len(parts) >= 4:
                            used = int(float(parts[1]))
                            total = int(float(parts[2]))
                            util = int(float(parts[3]))
                            cards.append(
                                {
                                    "node": node_name,
                                    "name": parts[0],
                                    "memory_used_mib": used,
                                    "memory_total_mib": total,
                                    "memory_free_mib": max(0, total - used),
                                    "utilization_pct": util,
                                    "source": f"k8s:{pod_name}",
                                }
                            )
        except Exception as exc:
            return {
                "ok": False,
                "status": "unavailable",
                "error": str(exc),
                "cards": [],
            }

    if cards:
        return {"ok": True, "status": "ok", "cards": cards}

    local_smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=5,
    )
    if local_smi.returncode == 0 and local_smi.stdout.strip():
        for row in local_smi.stdout.strip().splitlines():
            parts = [p.strip() for p in row.split(",")]
            if len(parts) >= 4:
                used = int(float(parts[1]))
                total = int(float(parts[2]))
                util = int(float(parts[3]))
                cards.append(
                    {
                        "node": "local",
                        "name": parts[0],
                        "memory_used_mib": used,
                        "memory_total_mib": total,
                        "memory_free_mib": max(0, total - used),
                        "utilization_pct": util,
                        "source": "local:nvidia-smi",
                    }
                )
        return {"ok": True, "status": "ok", "cards": cards}

    return {
        "ok": False,
        "status": "unavailable",
        "error": (proc.stderr or "No GPU metrics available").strip(),
        "cards": [],
    }


def collect_git() -> dict[str, Any]:
    """Read-only git status, branch, recent commits, and diff stat."""
    branch_proc = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=5)
    head_proc = _run(["git", "rev-parse", "HEAD"], timeout=5)
    if branch_proc.returncode != 0 or head_proc.returncode != 0:
        return {
            "ok": False,
            "status": "unavailable",
            "error": (branch_proc.stderr or head_proc.stderr or "git unavailable").strip(),
        }

    branch = branch_proc.stdout.strip()
    head_sha = head_proc.stdout.strip()
    target_ref = "freestyle-cartridges"
    check_ref = _run(["git", "rev-parse", "--verify", target_ref], timeout=5)
    if check_ref.returncode != 0:
        target_ref = "HEAD"

    oneline_proc = _run(["git", "log", "--oneline", "-25", target_ref], timeout=10)
    structured_proc = _run(
        ["git", "log", "--format=%H%x1f%h%x1f%cI%x1f%an%x1f%s", "-25", target_ref],
        timeout=10,
    )
    diff_proc = _run(["git", "diff", "--stat"], timeout=10)
    status_proc = _run(["git", "status", "--porcelain"], timeout=10)

    commits: list[dict[str, str]] = []
    for line in structured_proc.stdout.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 5:
            commits.append(
                {
                    "sha": parts[0],
                    "short_sha": parts[1],
                    "date": parts[2],
                    "author": parts[3],
                    "subject": parts[4],
                }
            )

    dirty_lines = [ln for ln in status_proc.stdout.splitlines() if ln.strip()]
    untracked = sum(1 for ln in dirty_lines if ln.startswith("??"))
    modified = len(dirty_lines) - untracked

    return {
        "ok": True,
        "status": "ok",
        "branch": branch,
        "head_sha": head_sha,
        "short_sha": head_sha[:7],
        "oneline": oneline_proc.stdout.splitlines(),
        "commits": commits,
        "diff_stat": diff_proc.stdout.strip(),
        "dirty": {
            "modified": modified,
            "untracked": untracked,
            "total": len(dirty_lines),
        },
    }


def collect_tests(head_sha: str | None) -> dict[str, Any]:
    """Run ``uv run pytest tests/unit -q`` only when HEAD changes; cache result."""
    if head_sha and PYTEST_CACHE_PATH.exists():
        try:
            cached = json.loads(PYTEST_CACHE_PATH.read_text())
            if cached.get("head_sha") == head_sha and cached.get("ok") is True:
                cached["cached"] = True
                return cached
        except Exception:
            pass

    proc = _run(["uv", "run", "pytest", "tests/unit", "-q"], timeout=180)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    summary_line = ""
    for line in reversed(output.splitlines()):
        if re.search(r"\b(passed|failed|error|skipped)\b", line):
            summary_line = line.strip()
            break

    passed = _match_int(summary_line, r"(\d+)\s+passed")
    failed = _match_int(summary_line, r"(\d+)\s+failed")
    skipped = _match_int(summary_line, r"(\d+)\s+skipped")
    duration_match = re.search(r"in\s+([\d.]+)s", summary_line)
    duration_s = float(duration_match.group(1)) if duration_match else None

    result = {
        "ok": proc.returncode == 0,
        "status": "ok" if proc.returncode == 0 else "failed",
        "head_sha": head_sha,
        "timestamp": _now_utc().isoformat(),
        "returncode": proc.returncode,
        "summary": summary_line or f"exit {proc.returncode}",
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "duration_s": duration_s,
        "cached": False,
    }
    if not result["ok"]:
        result["error"] = "\n".join(output.splitlines()[-15:])
    if head_sha:
        try:
            SITE_DIR.mkdir(parents=True, exist_ok=True)
            PYTEST_CACHE_PATH.write_text(json.dumps(result, indent=2) + "\n")
        except Exception:
            pass
    return result


def _match_int(text: str, pattern: str) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0


def _is_chain_check(run_dir: Path, metrics_data: dict[str, Any], meta_data: dict[str, Any]) -> bool:
    if (meta_data.get("config") or {}).get("chain_check") is True:
        return True
    if metrics_data.get("chain_check") is True:
        return True
    haystack = " ".join(
        [
            run_dir.name,
            str(metrics_data.get("experiment", "")),
            str(metrics_data.get("gt_path", "")),
        ]
    ).lower()
    return "chain_check" in haystack or "chain-check" in haystack


def _classify_gt_source(metrics_data: dict[str, Any], meta_data: dict[str, Any]) -> tuple[str, str]:
    gt_source = metrics_data.get("gt_source")
    gt_note = metrics_data.get("gt_note")
    if gt_source or gt_note:
        combined = f"{gt_source or ''} {gt_note or ''}".lower()
        if "human" in combined and "not human" not in combined and "model" not in combined:
            return "human", "vs human GT"
        if any(tok in combined for tok in ("model", "bootstrap", "auto")):
            return "model_labels", "vs model labels (not GT)"

    data_cfg = (meta_data.get("config") or {}).get("data") or {}
    for candidate in (metrics_data.get("gt_path"), data_cfg.get("test")):
        if not candidate:
            continue
        p = str(candidate).lower()
        if "labels/" in p or "bootstrap" in p:
            return "model_labels", "vs model labels (not GT)"
        if ("data/gt/" in p or p.startswith("gt/")) and "_fake" not in p and "_chaincheck" not in p:
            return "human", "vs human GT"

    if str(data_cfg.get("label_source") or "").lower() == "gt":
        return "human", "vs human GT"
    return "model_labels", "vs model labels (not GT)"


def collect_experiments() -> dict[str, Any]:
    """Collect experiment metrics from runs/*/metrics.json, smoke probes, and RESULTS.md."""
    runs_dir = REPO_ROOT / "runs"
    metrics_runs: list[dict[str, Any]] = []
    smoke_runs: list[dict[str, Any]] = []
    probe_summary: dict[str, Any] | None = None

    if runs_dir.exists():
        for metrics_file in sorted(runs_dir.glob("*/metrics.json")):
            run_dir = metrics_file.parent
            try:
                data = json.loads(metrics_file.read_text())
                meta_file = run_dir / "meta.json"
                meta: dict[str, Any] = {}
                if meta_file.exists():
                    try:
                        meta = json.loads(meta_file.read_text())
                    except Exception:
                        meta = {}
                if _is_chain_check(run_dir, data, meta):
                    continue

                loc = data.get("localization", {})
                ident = data.get("identification", {})
                cnt = data.get("counting", {})
                gt_kind, gt_badge = _classify_gt_source(data, meta)

                peak_bytes = meta.get("peak_gpu_bytes")
                nvidia_smi_raw = meta.get("nvidia_smi")
                nvidia_smi_mib: int | None = None
                if isinstance(nvidia_smi_raw, (int, float)):
                    nvidia_smi_mib = int(nvidia_smi_raw)
                elif isinstance(nvidia_smi_raw, str) and nvidia_smi_raw.strip().isdigit():
                    nvidia_smi_mib = int(nvidia_smi_raw.strip())

                peak_gpu_mib: int | None = None
                if isinstance(peak_bytes, (int, float)) and peak_bytes > 0:
                    peak_gpu_mib = round(peak_bytes / (1024 * 1024))
                elif nvidia_smi_mib is not None and nvidia_smi_mib > 0:
                    peak_gpu_mib = nvidia_smi_mib

                job_wall = meta.get("job_wall") or {}
                timings = meta.get("timings") or {}
                wall_seconds: float | None = None
                if isinstance(job_wall.get("job_total"), (int, float)):
                    wall_seconds = float(job_wall["job_total"])
                elif isinstance(job_wall.get("command"), (int, float)):
                    wall_seconds = float(job_wall["command"])
                elif timings:
                    vals = [float(v) for v in timings.values() if isinstance(v, (int, float))]
                    if vals:
                        wall_seconds = sum(vals)

                metrics_runs.append(
                    {
                        "id": data.get("experiment") or run_dir.name,
                        "dir": run_dir.name,
                        "ap50": loc.get("ap50"),
                        "ap50_95": loc.get("ap50_95"),
                        "precision": loc.get("precision"),
                        "recall": loc.get("recall"),
                        "f1": loc.get("f1"),
                        "top1_acc": ident.get("top1_acc"),
                        "macro_f1": ident.get("macro_f1"),
                        "total_count_mae": cnt.get("total_count_mae"),
                        "exact_inventory_rate": cnt.get("exact_inventory_rate"),
                        "score_thr": data.get("score_thr"),
                        "gt_path": data.get("gt_path"),
                        "gt_source": data.get("gt_source"),
                        "gt_note": data.get("gt_note"),
                        "gt_kind": gt_kind,
                        "gt_badge": gt_badge,
                        "peak_gpu_bytes": peak_bytes,
                        "peak_gpu_mib": peak_gpu_mib,
                        "nvidia_smi_mib": nvidia_smi_mib,
                        "wall_seconds": wall_seconds,
                        "wall_duration": _format_duration(wall_seconds),
                    }
                )
            except Exception as exc:
                metrics_runs.append({"id": run_dir.name, "error": str(exc)})

        for smoke_file in sorted(runs_dir.glob("*/smoke.json")):
            try:
                sdata = json.loads(smoke_file.read_text())
                dev = sdata.get("device", {})
                matmul = sdata.get("matmul", {})
                fp16 = (matmul.get("float16") or {}).get("tflops")
                bf16 = (matmul.get("bfloat16") or {}).get("tflops")
                smoke_runs.append(
                    {
                        "job": sdata.get("job") or smoke_file.parent.name,
                        "gpu": dev.get("name"),
                        "capability": dev.get("capability"),
                        "fp16_tflops": fp16,
                        "bf16_tflops": bf16,
                        "peak_mem_mib": sdata.get("peak_memory_mib"),
                        "finished": sdata.get("finished"),
                    }
                )
            except Exception:
                pass

        probe_file = runs_dir / "probe-collective" / "results.json"
        if probe_file.exists():
            try:
                pdata = json.loads(probe_file.read_text())
                tiny = pdata.get("tiny_probes", {})
                probe_summary = {
                    "started_at": pdata.get("started_at"),
                    "updated_at": pdata.get("updated_at"),
                    "models_count": len(pdata.get("models_list", [])),
                    "tiny_ok": sum(1 for v in tiny.values() if isinstance(v, dict) and v.get("ok")),
                    "tiny_total": len(tiny),
                    "detect_runs": len(pdata.get("detect", [])),
                    "identify_runs": len(pdata.get("identify", [])),
                    "errors": len(pdata.get("errors", [])),
                }
            except Exception:
                pass

    results_md_path = REPO_ROOT / "docs/freestyle/RESULTS.md"
    results_md = None
    if results_md_path.exists():
        results_md = {
            "path": "docs/freestyle/RESULTS.md",
            "content": results_md_path.read_text(),
        }

    return {
        "ok": True,
        "status": "ok",
        "metrics_runs": metrics_runs,
        "smoke_runs": smoke_runs,
        "collective_probe": probe_summary,
        "results_md": results_md,
    }


def _safe_step(name: str, fn: Any, *args: Any) -> dict[str, Any]:
    try:
        return fn(*args)
    except Exception as exc:
        return {
            "ok": False,
            "status": "unavailable",
            "error": f"{name} failed: {exc}",
        }


def collect_all() -> dict[str, Any]:
    now = _now_utc()
    local_now = datetime.now().astimezone()

    git_info = _safe_step("git", collect_git)
    head_sha = git_info.get("head_sha") if git_info.get("ok") else None

    payload: dict[str, Any] = {
        "version": 1,
        "timestamp": now.isoformat(),
        "local_timestamp": local_now.strftime("%Y-%m-%d %H:%M %Z"),
        "pi_workers": _safe_step("pi_workers", collect_pi_workers),
        "external_agents": _safe_step("external_agents", collect_external_agents),
        "k8s": _safe_step("k8s", collect_k8s),
        "gpus": _safe_step("gpus", collect_gpus),
        "git": git_info,
        "tests": _safe_step("tests", collect_tests, head_sha),
        "experiments": _safe_step("experiments", collect_experiments),
    }
    return payload


def main() -> int:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    status = collect_all()
    tmp_path = SITE_DIR / f".status.{os.getpid()}.tmp"
    tmp_path.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(tmp_path, STATUS_PATH)
    print(f"Wrote {STATUS_PATH} at {status['local_timestamp']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
