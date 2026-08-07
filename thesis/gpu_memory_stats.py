"""CUDA peak memory helpers for eval generate timing."""

from __future__ import annotations

import subprocess
from typing import Any


def cuda_is_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def reset_peak_gpu_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _bytes_to_gib(n: int) -> float:
    return round(n / (1024**3), 3)


def collect_gpu_memory_snapshot(*, include_nvidia_smi: bool = True) -> dict[str, Any]:
    """Peak and current CUDA memory since last reset_peak_gpu_memory()."""
    try:
        import torch
    except Exception as exc:
        return {"cuda_available": False, "error": str(exc)}

    if not torch.cuda.is_available():
        return {"cuda_available": False}

    torch.cuda.synchronize()
    peak_alloc = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    cur_alloc = int(torch.cuda.memory_allocated())
    cur_reserved = int(torch.cuda.memory_reserved())

    out: dict[str, Any] = {
        "cuda_available": True,
        "device_index": int(torch.cuda.current_device()),
        "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_gib": _bytes_to_gib(peak_alloc),
        "peak_reserved_gib": _bytes_to_gib(peak_reserved),
        "current_allocated_bytes": cur_alloc,
        "current_reserved_bytes": cur_reserved,
        "current_allocated_gib": _bytes_to_gib(cur_alloc),
        "current_reserved_gib": _bytes_to_gib(cur_reserved),
    }

    if include_nvidia_smi:
        smi = _nvidia_smi_snapshot()
        if smi:
            out["nvidia_smi"] = smi
    return out


def _nvidia_smi_snapshot() -> dict[str, Any] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    if not out:
        return None
    # index, name, total, used, free (MiB)
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) < 5:
        return None
    return {
        "gpu_index": parts[0],
        "gpu_name": parts[1],
        "memory_total_mib": int(float(parts[2])),
        "memory_used_mib": int(float(parts[3])),
        "memory_free_mib": int(float(parts[4])),
    }


def merge_memory_phases(
    *,
    after_load: dict[str, Any],
    after_generate: dict[str, Any],
    job_total: dict[str, Any],
) -> dict[str, Any]:
    """Compact memory block for timing.json."""
    if not job_total.get("cuda_available"):
        return {"cuda_available": False}

    block: dict[str, Any] = {
        "cuda_available": True,
        "device_name": job_total.get("device_name"),
        "after_load_peak_allocated_gib": after_load.get("peak_allocated_gib"),
        "after_load_peak_reserved_gib": after_load.get("peak_reserved_gib"),
        "inference_peak_allocated_gib": after_generate.get("peak_allocated_gib"),
        "inference_peak_reserved_gib": after_generate.get("peak_reserved_gib"),
        "job_peak_allocated_gib": job_total.get("peak_allocated_gib"),
        "job_peak_reserved_gib": job_total.get("peak_reserved_gib"),
        "end_current_allocated_gib": job_total.get("current_allocated_gib"),
        "end_current_reserved_gib": job_total.get("current_reserved_gib"),
    }
    smi = job_total.get("nvidia_smi")
    if smi:
        block["nvidia_smi_end_used_mib"] = smi.get("memory_used_mib")
        block["nvidia_smi_total_mib"] = smi.get("memory_total_mib")
    return block
