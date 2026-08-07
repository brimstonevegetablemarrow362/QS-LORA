"""
Experiment run logging for thesis RepLiQA pipelines (timings + provenance).

Writes under ``<run_dir>/experiment/``:
  - ``run_manifest.json`` — full config, data paths, durations, trainer stats
  - ``spans.jsonl`` — append-only timed steps
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision(cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or Path.cwd()),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _env_snapshot() -> dict[str, str]:
    keys = (
        "HOSTNAME",
        "SLURM_JOB_ID",
        "SLURM_NODELIST",
        "CUDA_VISIBLE_DEVICES",
        "DOMAIN_BASE_MODEL_ID",
    )
    return {k: os.environ[k] for k in keys if os.environ.get(k)}


@dataclass
class ExperimentLogger:
    run_dir: Path
    baseline: str
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    started_at: str = field(default_factory=utc_iso)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir).expanduser().resolve()
        self.exp_dir = self.run_dir / "experiment"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.exp_dir / "run_manifest.json"
        self.spans_path = self.exp_dir / "spans.jsonl"
        self._spans: list[dict[str, Any]] = []
        self._manifest: dict[str, Any] = {
            "schema": "thesis_experiment_run/v1",
            "run_id": self.run_id,
            "baseline": self.baseline,
            "started_at": self.started_at,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "git_head": _git_revision(Path(__file__).resolve().parent.parent),
            "env": _env_snapshot(),
            "spans": [],
            "artifacts": {},
            "hyperparameters": {},
            "data": {},
            "timing": {"total_wall_s": None, "steps": {}},
            "trainer": {},
            "status": "running",
        }
        self._flush_manifest()

    def _flush_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self._manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def set_section(self, section: str, data: dict[str, Any]) -> None:
        self._manifest[section] = data
        self._flush_manifest()

    def update_section(self, section: str, data: dict[str, Any]) -> None:
        cur = self._manifest.get(section)
        if isinstance(cur, dict):
            cur.update(data)
        else:
            self._manifest[section] = data
        self._flush_manifest()

    def add_artifact(self, name: str, path: Path | str) -> None:
        self._manifest.setdefault("artifacts", {})[name] = str(path)
        self._flush_manifest()

    def emit_span(
        self,
        step: str,
        duration_s: float,
        *,
        status: str = "ok",
        extra: dict[str, Any] | None = None,
    ) -> None:
        rec: dict[str, Any] = {
            "step": step,
            "duration_s": round(float(duration_s), 3),
            "status": status,
            "ts_end": utc_iso(),
        }
        if extra:
            rec["extra"] = extra
        self._spans.append(rec)
        with self.spans_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        steps = self._manifest.setdefault("timing", {}).setdefault("steps", {})
        if step not in steps:
            steps[step] = {"count": 0, "total_s": 0.0, "max_s": 0.0}
        steps[step]["count"] += 1
        steps[step]["total_s"] = round(steps[step]["total_s"] + duration_s, 3)
        steps[step]["max_s"] = round(max(steps[step]["max_s"], duration_s), 3)
        self._flush_manifest()

    @contextmanager
    def span(self, step: str, **extra: Any) -> Iterator[None]:
        t0 = time.perf_counter()
        status = "ok"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            self.emit_span(step, time.perf_counter() - t0, status=status, extra=extra or None)

    def ingest_trainer_state(self, adapter_dir: Path) -> None:
        """Read HuggingFace ``trainer_state.json`` if present."""
        state_path = adapter_dir / "trainer_state.json"
        if not state_path.is_file():
            for ckpt in sorted(adapter_dir.glob("checkpoint-*")):
                alt = ckpt / "trainer_state.json"
                if alt.is_file():
                    state_path = alt
                    break
        if not state_path.is_file():
            return
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        metrics: dict[str, Any] = {}
        if "train_runtime" in state:
            metrics["train_runtime_s"] = round(float(state["train_runtime"]), 3)
        if "train_samples_per_second" in state:
            metrics["train_samples_per_second"] = state["train_samples_per_second"]
        if "epoch" in state:
            metrics["epochs_completed"] = state["epoch"]
        log_hist = state.get("log_history") or []
        if log_hist:
            metrics["last_log"] = log_hist[-1]
        self._manifest["trainer"] = {
            "trainer_state_path": str(state_path),
            "metrics": metrics,
        }
        self._flush_manifest()

    def finalize(self, *, status: str, started_wall: float) -> dict[str, Any]:
        total = time.perf_counter() - started_wall
        self._manifest["finished_at"] = utc_iso()
        self._manifest["status"] = status
        self._manifest.setdefault("timing", {})["total_wall_s"] = round(total, 3)
        self._flush_manifest()
        return self._manifest
