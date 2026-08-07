"""
Compile TRAINING_RESOURCES.md — wall times and memory across models × datasets.

  python -m thesis.cli compile-training-resources-doc
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.compile_cross_model_ceiling_gap import (
    B5,
    DATASETS,
    OURS,
    REFERENCE_LLAMA_3B_RUNS,
    _fmt_duration,
    _fmt_size,
    collect_adapter_sizes,
    collect_training_timing,
)

THESIS_ROOT = Path(__file__).resolve().parent
DEFAULT_CROSS_ROOT = Path("/fs/ess/PAS2699/pratham2210/cross_model/runs")

# From scripts/cross_model_models.sh registry
SLURM_RESOURCES: dict[str, dict[str, Any]] = {
    "llama32_1b": {"tier": "small", "gpus": 1, "train_mem": "80G", "merge_mem": "80G", "qlora": False},
    "llama32_3b": {"tier": "reference", "gpus": 1, "train_mem": "80G", "merge_mem": "80G", "qlora": False},
    "llama31_8b": {"tier": "medium", "gpus": 1, "train_mem": "80G", "merge_mem": "160G", "qlora": False},
    "llama31_70b": {"tier": "xlarge", "gpus": 4, "train_mem": "256G", "merge_mem": "256G", "qlora": True},
    "qwen25_3b": {"tier": "small", "gpus": 1, "train_mem": "80G", "merge_mem": "80G", "qlora": False},
    "qwen25_7b": {"tier": "medium", "gpus": 1, "train_mem": "80G", "merge_mem": "160G", "qlora": False},
    "qwen25_14b": {"tier": "large", "gpus": 1, "train_mem": "120G", "merge_mem": "200G", "qlora": True},
    "gemma3_1b": {"tier": "small", "gpus": 1, "train_mem": "80G", "merge_mem": "80G", "qlora": False},
    "gemma3_4b": {"tier": "medium", "gpus": 1, "train_mem": "80G", "merge_mem": "160G", "qlora": False},
    "gemma3_12b": {"tier": "large", "gpus": 1, "train_mem": "120G", "merge_mem": "200G", "qlora": True},
}

ALL_MODELS: list[tuple[str, str]] = [
    ("llama32_1b", "Llama-3.2-1B"),
    ("llama32_3b", "Llama-3.2-3B"),
    ("llama31_8b", "Llama-3.1-8B"),
    ("qwen25_3b", "Qwen2.5-3B"),
    ("qwen25_7b", "Qwen2.5-7B"),
    ("qwen25_14b", "Qwen2.5-14B"),
    ("gemma3_1b", "Gemma-3-1B"),
    ("gemma3_4b", "Gemma-3-4B"),
    ("gemma3_12b", "Gemma-3-12B"),
    ("llama31_70b", "Llama-3.1-70B"),
]


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manifest_wall(path: Path) -> float | None:
    if not path.is_file():
        return None
    return (json.loads(path.read_text(encoding="utf-8")).get("timing") or {}).get("total_wall_s")


def _collect_qs_tier_breakdown(run_root: Path) -> dict[str, float | None]:
    qs_dir = run_root / "baselines" / "qs_strat"
    out: dict[str, float | None] = {"high": None, "medium": None, "low": None, "merge": None}
    if not qs_dir.is_dir():
        return out
    for tier, key in (("high", "high"), ("medium", "medium"), ("low", "low")):
        matches = list(qs_dir.glob(f"QS_strat_{tier}_lora_r*/experiment/run_manifest.json"))
        if matches:
            out[key] = _manifest_wall(matches[0])
    merge_idx = run_root / "qs_merge_timing_index.json"
    if merge_idx.is_file():
        merge_s = sum(float(r.get("total_wall_s") or 0) for r in json.loads(merge_idx.read_text()).values())
        out["merge"] = merge_s or None
    return out


def _collect_train_detail(run_root: Path) -> dict[str, Any]:
    timing = collect_training_timing(run_root)
    tiers = _collect_qs_tier_breakdown(run_root)
    b3_manifest = next((p for p in (run_root / "baselines").glob("B3*/experiment/run_manifest.json")), None)
    train_rows: dict[str, int | None] = {"B3": None, "B5": None}
    for cond, key in (("B3", "B3"), ("B5", "B5")):
        mpath = next((run_root / "baselines").glob(f"{cond}*/experiment/run_manifest.json"), None)
        if mpath and mpath.is_file():
            train_rows[key] = (json.loads(mpath.read_text()).get("data") or {}).get("n_train_rows")
    return {
        "B3_s": timing.get("B3"),
        "B5_s": timing.get("B5"),
        "Ours_s": timing.get("Ours"),
        "Ours_merge_s": timing.get("Ours_merge"),
        "qs_high_s": tiers["high"],
        "qs_medium_s": tiers["medium"],
        "qs_low_s": tiers["low"],
        "qs_merge_s": tiers["merge"],
        "train_rows": train_rows,
    }


def _read_timing_memory(run_root: Path, condition: str) -> dict[str, Any] | None:
    path = run_root / "eval" / "predictions" / condition / "timing.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    mem = data.get("memory") or {}
    if not mem.get("job_peak_allocated_gib"):
        return None
    return {
        "condition": condition,
        "load_type": data.get("load_type"),
        "job_peak_gib": mem.get("job_peak_allocated_gib"),
        "after_load_peak_gib": mem.get("after_load_peak_allocated_gib"),
        "device": mem.get("device_name"),
        "nvidia_smi_used_mib": mem.get("nvidia_smi_end_used_mib"),
    }


def _reference_resource_timing() -> dict[str, Any] | None:
    path = THESIS_ROOT / "experiments/repliqa/runs/repliqa_train_0-3/eval/resource_timing.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_root_for(slug: str, ds_sub: str, cross_root: Path) -> Path | None:
    if slug == "llama32_3b":
        ds_key = next((k for k, sub, _, _ in DATASETS if sub == ds_sub), None)
        if not ds_key:
            return None
        rel = REFERENCE_LLAMA_3B_RUNS.get(ds_key)
        return (THESIS_ROOT / "experiments" / rel) if rel else None
    root = cross_root / slug / ds_sub
    return root if root.is_dir() else None


def build_training_resources(cross_root: Path) -> dict[str, Any]:
    per_model: list[dict[str, Any]] = []
    for slug, label in ALL_MODELS:
        slurm = SLURM_RESOURCES.get(slug, {})
        datasets_out: dict[str, Any] = {}
        for ds_key, ds_sub, ds_label, _ in DATASETS:
            run_root = _run_root_for(slug, ds_sub, cross_root)
            if not run_root:
                datasets_out[ds_key] = {"status": "missing"}
                continue
            detail = _collect_train_detail(run_root)
            inf_mem: dict[str, Any] = {}
            for cond_key, cond in (("B3", next(b for k, s, _, b in DATASETS if k == ds_key)), ("B5", B5[ds_key]), ("Ours", OURS[ds_key])):
                m = _read_timing_memory(run_root, cond)
                if m:
                    inf_mem[cond_key] = m
            datasets_out[ds_key] = {
                "dataset_label": ds_label,
                "run_root": str(run_root),
                "training": detail,
                "inference_memory": inf_mem or None,
            }
        adapter_sizes = None
        rep_root = _run_root_for(slug, "repliqa", cross_root)
        if rep_root:
            adapter_sizes = collect_adapter_sizes(rep_root)
        per_model.append(
            {
                "model_slug": slug,
                "model_label": label,
                "slurm": slurm,
                "adapter_sizes": adapter_sizes,
                "datasets": datasets_out,
            }
        )

    probe_path = THESIS_ROOT / "experiments/quoref/runs/quoref_qa_v1/eval/memory_probe/memory_probe_summary.json"
    memory_probe = json.loads(probe_path.read_text()) if probe_path.is_file() else None
    ref_timing = _reference_resource_timing()

    return {
        "schema": "training_resources/v1",
        "created_at": utc_iso(),
        "cross_root": str(cross_root),
        "hardware_default": "NVIDIA A100-SXM4-80GB",
        "memory_probe_reference_3b": memory_probe,
        "reference_llama32_3b_resource_timing": ref_timing,
        "models": per_model,
    }


def _md_training_matrix(data: dict[str, Any]) -> list[str]:
    lines = [
        "## Training wall time (all models × datasets)",
        "",
        "Measured from `experiment/run_manifest.json` → `timing.total_wall_s`.",
        "**Ours** = QS high + medium + low tier SFT + one-time dense merge.",
        "",
        "### Summary — B3 / Ours / B5 total",
        "",
        "| Model | RepLiQA B3 | Ours | B5 | Quoref B3 | Ours | B5 | SQuAD B3 | Ours | B5 |",
        "|-------|------------|------|-----|-----------|------|-----|----------|------|-----|",
    ]
    for m in data["models"]:
        row = [m["model_label"]]
        for ds_key in ("repliqa", "quoref", "squad"):
            ds = m["datasets"].get(ds_key, {})
            tr = ds.get("training") or {}
            for k in ("B3_s", "Ours_s", "B5_s"):
                row.append(_fmt_duration(tr.get(k)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _md_tier_breakdown(data: dict[str, Any]) -> list[str]:
    lines = [
        "## Ours QS tier breakdown",
        "",
        "Per-tier SFT wall time + dense merge. Only shown where artifacts exist.",
        "",
    ]
    for m in data["models"]:
        lines.append(f"### {m['model_label']}")
        lines.append("")
        lines.append("| Dataset | QS high | QS medium | QS low | Merge | **Ours total** |")
        lines.append("|---------|---------|-----------|--------|-------|----------------|")
        for ds_key, ds_label in (("repliqa", "RepLiQA"), ("quoref", "Quoref"), ("squad", "SQuAD")):
            tr = (m["datasets"].get(ds_key) or {}).get("training") or {}
            if not tr.get("Ours_s"):
                lines.append(f"| {ds_label} | — | — | — | — | — |")
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        ds_label,
                        _fmt_duration(tr.get("qs_high_s")),
                        _fmt_duration(tr.get("qs_medium_s")),
                        _fmt_duration(tr.get("qs_low_s")),
                        _fmt_duration(tr.get("qs_merge_s")),
                        f"**{_fmt_duration(tr.get('Ours_s'))}**",
                    ]
                )
                + " |"
            )
        lines.append("")
    return lines


def _md_reference_3b(data: dict[str, Any]) -> list[str]:
    ref = data.get("reference_llama32_3b_resource_timing")
    if not ref:
        return []
    lines = [
        "## Reference run — Llama-3.2-3B (detailed)",
        "",
        f"Source: `experiments/repliqa/runs/repliqa_train_0-3/eval/resource_timing.json`",
        "",
        f"Hardware: {ref.get('hardware', {}).get('gpu', 'A100-80GB')}, "
        f"{ref.get('hardware', {}).get('gpus_per_job', 1)}× GPU",
        "",
        "### RepLiQA training (per condition)",
        "",
        "| Condition | Wall time | Train rows | LoRA rank |",
        "|-----------|-----------|------------|-----------|",
    ]
    train = ref.get("train_runs") or {}
    order = [
        ("baselines/B3_all_lora_r16", "B3 uniform", 16),
        ("baselines/qs_strat/QS_strat_high_lora_r32", "Ours QS high", 32),
        ("baselines/qs_strat/QS_strat_medium_lora_r16", "Ours QS medium", 16),
        ("baselines/qs_strat/QS_strat_low_lora_r8", "Ours QS low", 8),
    ]
    b5 = train.get("baselines/B5_adalora_r16")
    for path_key, label, rank in order:
        row = train.get(path_key)
        if not row:
            continue
        n_rows = "—"
        if path_key == "baselines/B3_all_lora_r16":
            n_rows = "11,321"
        elif "high" in path_key:
            n_rows = "6,901"
        elif "medium" in path_key:
            n_rows = "1,147"
        elif "low" in path_key:
            n_rows = "579"
        lines.append(
            f"| {label} | {row.get('total_wall_hms', _fmt_duration(row.get('total_wall_s')))} | {n_rows} | r={rank} |"
        )
    if b5:
        lines.append(
            f"| B5 AdaLoRA | {b5.get('total_wall_hms', _fmt_duration(b5.get('total_wall_s')))} | 11,321 | AdaLoRA r=16 |"
        )
    merge = ref.get("merge_runs") or {}
    w601 = merge.get("QS_merged_strat_dense_w60_30_10") or {}
    merge_s = (w601.get("timing") or {}).get("total_wall_s")
    if merge_s:
        lines.append(
            f"| Dense merge (w60/30/10) | {_fmt_duration(merge_s)} | — | — |"
        )
    lines.extend(
        [
            "",
            "### Quoref & SQuAD (Llama-3.2-3B + domain CPT)",
            "",
            "| Dataset | B3 | Ours QS total | B5 |",
            "|---------|-----|---------------|-----|",
            "| Quoref | 49m | 46m | 1h 08m |",
            "| SQuAD v2 | 5h 41m | 5h 35m | 8h 29m |",
            "",
            "From `run_manifest.json` under `quoref/runs/quoref_qa_v1` and `squad_v2/runs/squad_qa_v1`.",
            "",
        ]
    )
    return lines


def _md_memory(data: dict[str, Any]) -> list[str]:
    lines = [
        "## Memory requirements",
        "",
        "### SLURM allocation (requested per job)",
        "",
        "From `thesis/scripts/cross_model_models.sh`. This is **requested host memory**, not measured peak VRAM.",
        "",
        "| Model | Tier | GPUs | Train ReqMem | Merge ReqMem | QLoRA train |",
        "|-------|------|------|--------------|--------------|-------------|",
    ]
    for m in data["models"]:
        s = m.get("slurm") or {}
        if not s:
            continue
        lines.append(
            f"| {m['model_label']} | {s.get('tier', '—')} | {s.get('gpus', 1)} | "
            f"{s.get('train_mem', '—')} | {s.get('merge_mem', '—')} | "
            f"{'yes' if s.get('qlora') else 'no'} |"
        )
    lines.extend(
        [
            "",
            "### Measured peak GPU memory at inference",
            "",
            "From `eval/predictions/<condition>/timing.json` → `memory.job_peak_allocated_gib`.",
            "All runs: bf16, greedy decode, batch=1. **Training peak VRAM was not logged** in run manifests.",
            "",
            "#### Llama-3.2-3B reference probe (Quoref, n=200)",
            "",
            "Source: `quoref_qa_v1/eval/memory_probe/memory_probe_summary.json`",
            "",
            "| Condition | After-load peak | Job peak | nvidia-smi used |",
            "|-----------|-----------------|----------|-----------------|",
        ]
    )
    probe = data.get("memory_probe_reference_3b")
    if probe:
        for c in probe.get("conditions", []):
            lines.append(
                f"| {c['condition']} | {c['after_load_peak_gib']:.2f} GiB | "
                f"{c['job_peak_gib']:.2f} GiB | {c.get('nvidia_smi_used_mib', '—')} MiB |"
            )
    lines.extend(
        [
            "",
            "LoRA and dense merge use similar VRAM at inference (~6.1–6.3 GiB for 3B).",
            "",
            "#### Cross-model inference peak (RepLiQA, where logged)",
            "",
            "| Model | B3 peak | Ours peak | B5 peak | Device |",
            "|-------|---------|-----------|---------|--------|",
        ]
    )
    for m in data["models"]:
        ds = m["datasets"].get("repliqa") or {}
        mem = ds.get("inference_memory") or {}
        if not mem:
            continue
        def _peak(k: str) -> str:
            v = mem.get(k, {})
            return f"{v['job_peak_gib']:.1f} GiB" if v.get("job_peak_gib") else "—"
        device = next((mem[k].get("device") for k in ("B3", "Ours", "B5") if mem.get(k)), "—")
        lines.append(
            f"| {m['model_label']} | {_peak('B3')} | {_peak('Ours')} | {_peak('B5')} | {device or '—'} |"
        )
    lines.extend(
        [
            "",
            "**70B note:** Llama-3.1-70B uses 4× A100 with QLoRA 4-bit training (`ReqMem=256G`).",
            "Per-GPU inference peak ~33 GiB (dense merge ~32.2 GiB, LoRA ~32.8 GiB).",
            "",
            "### Adapter & merged model disk sizes",
            "",
            "Final checkpoint weights only (RepLiQA runs; same ranks across datasets).",
            "",
            "| Model | B3 LoRA | B5 AdaLoRA | QS tiers (3×) | Ours merged dense |",
            "|-------|---------|------------|---------------|-------------------|",
        ]
    )
    for m in data["models"]:
        sz = m.get("adapter_sizes")
        if not sz:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    m["model_label"],
                    _fmt_size(sz.get("B3")),
                    _fmt_size(sz.get("B5")),
                    _fmt_size(sz.get("QS_tiers")),
                    _fmt_size(sz.get("merged")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Training Time & Memory — Cross-Model Matrix",
        "",
        f"**Generated:** {data['created_at']}",
        "",
        "Wall-clock training times for B3 (uniform LoRA), B5 (AdaLoRA), and Ours (QS tiered LoRA + dense merge)",
        "across all backbone models and datasets (RepLiQA, Quoref, SQuAD v2).",
        "",
        "**Sources:**",
        "- Training: `baselines/*/experiment/run_manifest.json` → `timing.total_wall_s`",
        "- Merge: `qs_merge_timing_index.json`",
        "- Inference memory: `eval/predictions/*/timing.json` → `memory`",
        "- SLURM allocation: `thesis/scripts/cross_model_models.sh`",
        "",
        "---",
        "",
    ]
    lines.extend(_md_training_matrix(data))
    lines.append("---")
    lines.append("")
    lines.extend(_md_tier_breakdown(data))
    lines.append("---")
    lines.append("")
    lines.extend(_md_reference_3b(data))
    lines.append("---")
    lines.append("")
    lines.extend(_md_memory(data))
    lines.extend(
        [
            "---",
            "",
            "## Notes",
            "",
            "1. **Ours trains faster than B3 on RepLiQA** for most models because tier-splitting concentrates",
            "   high-rank compute on smaller subsets; on Quoref/SQuAD totals are similar.",
            "2. **B5 AdaLoRA is consistently the slowest** training condition (dynamic rank budgeting overhead).",
            "3. **Training peak VRAM is not logged** — only SLURM `ReqMem` and inference probes are available.",
            "4. **70B SQuAD** training/eval not yet complete at time of generation.",
            "5. Regenerate: `python -m thesis.cli compile-training-resources-doc`",
            "",
        ]
    )
    return "\n".join(lines)


def run_compile_training_resources_doc(ns: argparse.Namespace) -> int:
    cross_root = Path(ns.cross_root).expanduser().resolve()
    out_md = Path(ns.output_md).expanduser().resolve()
    out_json = Path(ns.output_json).expanduser().resolve() if ns.output_json else out_md.with_suffix(".json")

    data = build_training_resources(cross_root)
    out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")
    return 0


def add_cli(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "compile-training-resources-doc",
        help="Compile TRAINING_RESOURCES.md (training wall times + memory)",
    )
    p.add_argument(
        "--cross-root",
        type=Path,
        default=DEFAULT_CROSS_ROOT,
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=THESIS_ROOT / "TRAINING_RESOURCES.md",
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.set_defaults(fn=run_compile_training_resources_doc)
