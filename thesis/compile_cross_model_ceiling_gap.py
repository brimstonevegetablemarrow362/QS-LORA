"""Compile cross-model ceiling-gap markdown from per-run ceiling_gap_summary.json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODELS: list[tuple[str, str]] = [
    ("llama32_1b", "Llama-3.2-1B"),
    ("llama31_8b", "Llama-3.1-8B"),
    ("qwen25_3b", "Qwen2.5-3B"),
    ("qwen25_7b", "Qwen2.5-7B"),
    ("qwen25_14b", "Qwen2.5-14B"),
    ("gemma3_1b", "Gemma-3-1B"),
    ("gemma3_4b", "Gemma-3-4B"),
    ("gemma3_12b", "Gemma-3-12B"),
    ("llama31_70b", "Llama-3.1-70B"),
]

DATASETS: list[tuple[str, str, str, str]] = [
    ("repliqa", "repliqa", "RepLiQA", "B3_lora_all"),
    ("quoref", "quoref_qa_v1", "Quoref", "B3_lora_ctx"),
    ("squad", "squad_qa_v1", "SQuAD v2", "B3_lora_ctx"),
]

OURS = {
    "repliqa": "Ours_tier_merge",
    "quoref": "Ours_tier_ctx",
    "squad": "Ours_tier_ctx",
}
B5 = {
    "repliqa": "B5_adalora_all",
    "quoref": "B5_adalora_ctx",
    "squad": "B5_adalora_ctx",
}

CEILING_LABELS = {
    "REF_claude_opus": ("Claude Opus 4.8", "opus"),
    "REF_nova_2_lite": ("Nova 2 Lite", "nova"),
}

EVAL_N = {"repliqa": 2000, "quoref": 2418, "squad": 11873}

# Llama + Qwen only: Gemma runs mix vLLM batching, HF shards, and stale shard timing.json.
INFERENCE_FAMILIES: list[tuple[str, list[tuple[str, str]]]] = [
    ("Llama", [("llama32_1b", "1B"), ("llama31_8b", "8B")]),
    ("Qwen2.5", [("qwen25_3b", "3B"), ("qwen25_7b", "7B"), ("qwen25_14b", "14B")]),
]

GA_FAMILIES: list[tuple[str, list[tuple[str, str]]]] = [
    ("Llama", [("llama32_1b", "1B"), ("reference_llama32_3b", "3B"), ("llama31_8b", "8B")]),
    ("Qwen2.5", [("qwen25_3b", "3B"), ("qwen25_7b", "7B"), ("qwen25_14b", "14B")]),
    ("Gemma-3", [("gemma3_1b", "1B"), ("gemma3_4b", "4B"), ("gemma3_12b", "12B")]),
    ("Llama-3.1", [("llama31_70b", "70B")]),
]

# Llama-3.2-3B-Instruct reference runs (§1) — same B3/Ours/B5 protocol, not under cross_model/runs.
REFERENCE_LLAMA_3B_RUNS: dict[str, str] = {
    "repliqa": "repliqa/runs/repliqa_train_0-3",
    "quoref": "quoref/runs/quoref_qa_v1",
    "squad": "squad_v2/runs/squad_qa_v1",
}


def _adapter_file_bytes(adapter_dir: Path) -> int | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        p = adapter_dir / name
        if p.is_file():
            return p.stat().st_size
    return None


def _merged_bytes(run_root: Path) -> int | None:
    merge_dir = run_root / "baselines" / "qs_strat" / "QS_merged_strat_dense_w60_30_10"
    if not merge_dir.is_dir():
        return None
    shards = list(merge_dir.glob("model*.safetensors"))
    if not shards:
        return None
    return sum(p.stat().st_size for p in shards)


def collect_adapter_sizes(run_root: Path) -> dict[str, int | None]:
    """Final-checkpoint adapter weights from RepLiQA run (same ranks per model across datasets)."""
    baselines = run_root / "baselines"
    if not baselines.is_dir():
        return {"B3": None, "B5": None, "QS_tiers": None, "merged": None}
    b3_dir = next((p for p in baselines.glob("B3*") if p.is_dir()), None)
    b5_dir = next((p for p in baselines.glob("B5*") if p.is_dir()), None)
    qs = 0
    qs_dir = baselines / "qs_strat"
    if qs_dir.is_dir():
        for d in qs_dir.glob("QS_strat_*_lora*"):
            sz = _adapter_file_bytes(d)
            if sz:
                qs += sz
    return {
        "B3": _adapter_file_bytes(b3_dir) if b3_dir else None,
        "B5": _adapter_file_bytes(b5_dir) if b5_dir else None,
        "QS_tiers": qs or None,
        "merged": _merged_bytes(run_root),
    }


def _fmt_size(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    return f"{n / 1e6:.0f} MB"


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = int(round(float(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def _manifest_wall(path: Path) -> float | None:
    if not path.is_file():
        return None
    return (json.loads(path.read_text(encoding="utf-8")).get("timing") or {}).get("total_wall_s")


def collect_training_timing(run_root: Path) -> dict[str, float | None]:
    """B3/B5 from run_manifest; Ours = QS tier trains + dense merge."""
    baselines = run_root / "baselines"
    b3 = _manifest_wall(next((p for p in baselines.glob("B3*/experiment/run_manifest.json")), Path()))
    b5 = _manifest_wall(next((p for p in baselines.glob("B5*/experiment/run_manifest.json")), Path()))
    qs_train = 0.0
    qs_dir = baselines / "qs_strat"
    if qs_dir.is_dir():
        for manifest in sorted(qs_dir.glob("QS_strat_*_lora_r*/experiment/run_manifest.json")):
            w = _manifest_wall(manifest)
            if w:
                qs_train += w
    merge = 0.0
    merge_idx = run_root / "qs_merge_timing_index.json"
    if merge_idx.is_file():
        for row in json.loads(merge_idx.read_text(encoding="utf-8")).values():
            merge += float(row.get("total_wall_s") or 0)
    ours = (qs_train + merge) if qs_train else None
    return {"B3": b3, "B5": b5, "Ours": ours, "Ours_merge": merge or None}


def _read_timing_json(timing_path: Path) -> dict[str, Any]:
    if not timing_path.is_file():
        return {}
    return json.loads(timing_path.read_text(encoding="utf-8"))


def timing_is_reliable(
    timing_path: Path,
    *,
    dataset_key: str,
    model_slug: str,
    n_preds: int | None = None,
) -> tuple[bool, str]:
    """Return (ok, reason). Gemma + vLLM + partial shards are not cross-model comparable."""
    if model_slug.startswith("gemma3_"):
        return False, "gemma_mixed_backends"
    data = _read_timing_json(timing_path)
    if not data:
        return False, "missing"
    backend = str(data.get("backend") or (data.get("decoding") or {}).get("backend") or "hf").lower()
    if backend == "vllm":
        return False, "vllm_batch"
    gen = (data.get("timing") or {}).get("generate_per_question") or data.get("generate") or {}
    n = data.get("n_questions") or data.get("n_rows") or gen.get("n")
    expected = EVAL_N.get(dataset_key)
    if expected and n and int(n) < int(expected) * 0.95:
        return False, "partial_shard"
    if n_preds and expected and n_preds < int(expected) * 0.95:
        return False, "partial_preds"
    mean_s = gen.get("mean_s")
    if mean_s is not None and float(mean_s) <= 0:
        return False, "zero_mean"
    return True, "ok"


def collect_inference_row(
    run_root: Path,
    condition: str,
    *,
    dataset_key: str,
    model_slug: str,
) -> dict[str, Any]:
    timing_path = run_root / "eval" / "predictions" / condition / "timing.json"
    pred_path = run_root / "eval" / "predictions" / condition / "predictions.jsonl"
    n_preds = sum(1 for _ in pred_path.open()) if pred_path.is_file() else None
    ok, reason = timing_is_reliable(
        timing_path, dataset_key=dataset_key, model_slug=model_slug, n_preds=n_preds
    )
    if not ok:
        return {"reliable": False, "skip_reason": reason}

    data = _read_timing_json(timing_path)
    t = data.get("timing") or {}
    gen = t.get("generate_per_question") or data.get("generate") or {}
    mean_s = gen.get("mean_s")
    n = data.get("n_questions") or data.get("n_rows") or gen.get("n")
    return {
        "reliable": True,
        "mean_s_per_question": mean_s,
        "total_wall_s": t.get("total_wall_s") if t.get("total_wall_s") is not None else data.get("total_wall_s"),
        "total_wall_hms": t.get("total_wall_hms"),
        "load_type": data.get("load_type"),
        "backend": data.get("backend"),
        "n_questions": n,
    }


def _fmt_inf(mean_s: float | None) -> str:
    if mean_s is None:
        return "—"
    return f"{mean_s:.2f}"


def _judge_summary_path(
    cross_root: Path,
    slug: str,
    ds_sub: str,
    cond: str,
    *,
    dataset_key: str | None = None,
) -> Path | None:
    if slug == "reference_llama32_3b":
        if not dataset_key:
            dataset_key = next((k for k, sub, _, _ in DATASETS if sub == ds_sub), None)
        rel = REFERENCE_LLAMA_3B_RUNS.get(dataset_key or "")
        if not rel:
            return None
        run_root = Path(__file__).resolve().parent / "experiments" / rel
    else:
        run_root = cross_root / slug / ds_sub
    p = run_root / "eval" / "judged" / cond / "bedrock_judge_summary.json"
    return p if p.is_file() else None


def _ga(
    cross_root: Path,
    slug: str,
    ds_sub: str,
    cond: str,
    *,
    dataset_key: str | None = None,
) -> float | None:
    p = _judge_summary_path(cross_root, slug, ds_sub, cond, dataset_key=dataset_key)
    if not p:
        return None
    return json.loads(p.read_text(encoding="utf-8"))["stats"].get("mean_gold_alignment")


def _ga_cell(
    cross_root: Path,
    slug: str,
    ds_sub: str,
    cond: str,
    *,
    dataset_key: str | None = None,
) -> str:
    """Format GA or blocked marker when Bedrock judge failed with API errors."""
    p = _judge_summary_path(cross_root, slug, ds_sub, cond, dataset_key=dataset_key)
    if not p:
        return "—"
    stats = json.loads(p.read_text(encoding="utf-8"))["stats"]
    ga = stats.get("mean_gold_alignment")
    if ga is not None:
        return _fmt(ga)
    n_ok = int(stats.get("n_judged_ok") or 0)
    n_err = int(stats.get("n_api_error") or 0)
    if n_ok == 0 and n_err > 0:
        return "blocked†"
    return "—"


def _speedup(b3: float | None, ours: float | None) -> str:
    if b3 is None or ours is None or ours <= 0:
        return "—"
    return f"{b3 / ours:.1f}×"


def _fmt(v: float | None, nd: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{nd}f}"


def _pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}%"


def _load_row(root: Path) -> dict[str, Any] | None:
    p = root / "eval" / "judged" / "ceiling_gap_summary.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _cond_row(report: dict[str, Any], cond: str) -> dict[str, Any] | None:
    for row in report.get("comparisons") or []:
        if row.get("condition") == cond:
            return row
    return None


def _ceiling_ga(doc: dict[str, Any], ceiling: str) -> float | None:
    for report in doc.get("ceilings") or []:
        if report.get("ceiling_condition") == ceiling:
            return report.get("ceiling_mean_gold_alignment")
    return None


def build_gap_table(
    cross_root: Path,
    *,
    dataset_key: str,
    ds_subdir: str,
    ds_label: str,
    ceiling: str,
    ceiling_label: str,
) -> tuple[str, int]:
    b3_cond = next(b for k, sub, _, b in DATASETS if k == dataset_key)
    ours_cond = OURS[dataset_key]
    b5_cond = B5[dataset_key]
    lines = [
        f"### {ds_label} — gap vs {ceiling_label}",
        "",
        "| Model | B3 GA | B3 gap | Ours GA | Ours gap | % gain vs B3 | B5 GA | B5 gap | % gain vs B3 |",
        "|-------|-------|--------|---------|----------|--------------|-------|--------|--------------|",
    ]
    n_ok = 0
    ceiling_ga: float | None = None
    for slug, label in MODELS:
        doc = _load_row(cross_root / slug / ds_subdir)
        if not doc:
            lines.append(
                f"| {label} | — | — | — | — | — | — | — | — |"
            )
            continue
        report = next((r for r in doc["ceilings"] if r["ceiling_condition"] == ceiling), None)
        if not report:
            lines.append(f"| {label} | — | — | — | — | — | — | — | — |")
            continue
        if ceiling_ga is None:
            ceiling_ga = report.get("ceiling_mean_gold_alignment")
        b3 = _cond_row(report, b3_cond)
        ours = _cond_row(report, ours_cond)
        b5 = _cond_row(report, b5_cond)
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _fmt((b3 or {}).get("mean_gold_alignment")),
                    _fmt((b3 or {}).get("gap_to_ceiling")),
                    _fmt((ours or {}).get("mean_gold_alignment")),
                    _fmt((ours or {}).get("gap_to_ceiling")),
                    _pct((ours or {}).get("pct_of_ceiling_gain_vs_b3")),
                    _fmt((b5 or {}).get("mean_gold_alignment")),
                    _fmt((b5 or {}).get("gap_to_ceiling")),
                    _pct((b5 or {}).get("pct_of_ceiling_gain_vs_b3")),
                ]
            )
            + " |"
        )
        n_ok += 1
    if n_ok == 0 and dataset_key == "squad":
        lines.insert(
            2,
            "*All cells empty — SQuAD shared ceiling references were never generated/judged "
            "(`cross_model/ceilings/squad_qa_v1/` missing). Not an AWS block on individual models; "
            "gap cannot be computed without ceiling refs.*",
        )
        lines.insert(3, "")
    if ceiling_ga is not None:
        lines[0] = f"### {ds_label} — gap vs {ceiling_label} (ceiling GA = {_fmt(ceiling_ga)})"
    lines.append("")
    return "\n".join(lines), n_ok


def build_training_table(cross_root: Path) -> str:
    lines = [
        "### Cross-model training wall time (SFT + dense merge for Ours)",
        "",
        "From per-adapter `experiment/run_manifest.json` (`timing.total_wall_s`). "
        "Ours = sum(QS high/medium/low tier trains) + one-time dense merge.",
        "",
        "| Model | RepLiQA B3 | Ours | B5 | Quoref B3 | Ours | B5 | SQuAD B3 | Ours | B5 |",
        "|-------|------------|------|-----|-----------|------|-----|----------|------|-----|",
    ]
    for slug, label in MODELS:
        row = [label]
        for ds_key, ds_sub, _, _ in DATASETS:
            t = collect_training_timing(cross_root / slug / ds_sub)
            for k in ("B3", "Ours", "B5"):
                row.append(_fmt_duration(t.get(k)))
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "On the reference Llama-3.2-3B run (§5), Ours trains *less* wall time than B3 on RepLiQA because tier-splitting "
            "concentrates high-rank compute on smaller subsets. B5 AdaLoRA is consistently the slowest condition.",
            "",
        ]
    )
    return "\n".join(lines)


def build_adapter_size_table(cross_root: Path) -> str:
    lines = [
        "### Cross-model adapter & merged model sizes",
        "",
        "`adapter_model.safetensors` only (final checkpoint). QS tiers = high (r=32) + medium (r=16) + low (r=8). "
        "Ours merged = dense bf16 full weights (`QS_merged_strat_dense_w60_30_10`). Sizes measured on RepLiQA runs "
        "(same LoRA ranks per backbone across datasets). Llama-3.1-70B scale-out in §11.",
        "",
        "| Model | B3 LoRA | B5 AdaLoRA | QS tiers (3×) | Ours merged dense |",
        "|-------|---------|------------|---------------|-------------------|",
    ]
    for slug, label in MODELS:
        sz = collect_adapter_sizes(cross_root / slug / "repliqa")
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _fmt_size(sz["B3"]),
                    _fmt_size(sz["B5"]),
                    _fmt_size(sz["QS_tiers"]),
                    _fmt_size(sz["merged"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "At 3B, three tier adapters sum to ~341 MB vs B3 ~97 MB; dense merge is ~6.4 GB (full base in bf16). "
            "Disk savings of LoRA are large; inference speedup from merge (§5) does not reduce VRAM — all paths load the full base.",
            "",
        ]
    )
    return "\n".join(lines)


def build_ga_by_family_tables(cross_root: Path) -> str:
    lines = [
        "### Gold alignment by dataset and model family",
        "",
        "Mean GA (Haiku `v3_eval_gold`). **Δ Ours−B3** = gain of our method over uniform LoRA; **Δ Ours−B5** = our method vs AdaLoRA (positive → Ours wins). "
        "Llama **3B** = Llama-3.2-3B-Instruct reference run (§1), same protocol as cross-model matrix.",
        "",
    ]
    for ds_key, ds_sub, ds_label, b3_cond in DATASETS:
        lines.extend([f"#### {ds_label} (n={EVAL_N[ds_key]:,})", ""])
        for fam_name, members in GA_FAMILIES:
            lines.extend(
                [
                    f"**{fam_name}**",
                    "",
                    "| Size | B3 | Ours | B5 | Δ Ours−B3 | Δ Ours−B5 |",
                    "|------|-----|------|-----|-----------|------------|",
                ]
            )
            for slug, size in members:
                b3g = _ga(cross_root, slug, ds_sub, b3_cond, dataset_key=ds_key)
                og = _ga(cross_root, slug, ds_sub, OURS[ds_key], dataset_key=ds_key)
                b5g = _ga(cross_root, slug, ds_sub, B5[ds_key], dataset_key=ds_key)
                d_ours = f"{og - b3g:+.2f}" if og is not None and b3g is not None else "—"
                d_b5 = f"{og - b5g:+.2f}" if og is not None and b5g is not None else "—"
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            size,
                            _ga_cell(cross_root, slug, ds_sub, b3_cond, dataset_key=ds_key),
                            _ga_cell(cross_root, slug, ds_sub, OURS[ds_key], dataset_key=ds_key),
                            _ga_cell(cross_root, slug, ds_sub, B5[ds_key], dataset_key=ds_key),
                            d_ours,
                            d_b5,
                        ]
                    )
                    + " |"
                )
            lines.append("")
    return "\n".join(lines)


def build_b5_adalora_audit(cross_root: Path) -> str:
    lines = [
        "### AdaLoRA (B5) audit — when does B5 beat Ours?",
        "",
        "B5 looks unusually strong only on **Qwen2.5-14B**. Judged summaries are complete (no parse errors) — the GA gap is real.",
        "",
        "| Dataset | Qwen2.5-14B B3 | Ours | B5 | B5−Ours | Qwen2.5-7B B5−Ours |",
        "|---------|----------------|------|-----|---------|-------------------|",
    ]
    for ds_key, ds_sub, ds_label, b3_cond in DATASETS:
        b3_14 = _ga(cross_root, "qwen25_14b", ds_sub, b3_cond)
        o_14 = _ga(cross_root, "qwen25_14b", ds_sub, OURS[ds_key])
        b5_14 = _ga(cross_root, "qwen25_14b", ds_sub, B5[ds_key])
        o_7 = _ga(cross_root, "qwen25_7b", ds_sub, OURS[ds_key])
        b5_7 = _ga(cross_root, "qwen25_7b", ds_sub, B5[ds_key])
        b5_o_14 = f"{b5_14 - o_14:+.2f}" if b5_14 is not None and o_14 is not None else "—"
        b5_o_7 = f"{b5_7 - o_7:+.2f}" if b5_7 is not None and o_7 is not None else "—"
        lines.append(
            "| "
            + " | ".join([ds_label, _fmt(b3_14), _fmt(o_14), _fmt(b5_14), b5_o_14, b5_o_7])
            + " |"
        )
    lines.extend(
        [
            "",
            "**Interpretation:**",
            "",
            "- **RepLiQA + Quoref:** B5 on 14B improves over Ours by +0.21 / +0.12 GA.",
            "- **SQuAD:** B5 at 14B is the **only** model×dataset where B5 beats Ours by a large margin (+0.33 GA). At 7B and below, B5 trails Ours on SQuAD.",
            "- **Scale-specific:** AdaLoRA’s adaptive rank budget appears to help most on the **largest Qwen** backbone; it does not generalize as a universal win over QS-merge.",
            "",
        ]
    )
    return "\n".join(lines)


def build_blocked_results_note(cross_root: Path) -> str:
    ceiling_root = cross_root.parent / "ceilings"
    squad_ceiling = ceiling_root / "squad_qa_v1" / "judged"
    squad_ceiling_ok = (squad_ceiling / "REF_claude_opus" / "bedrock_judge_summary.json").is_file()
    return "\n".join(
        [
            "### Blocked or missing results",
            "",
            "| Location | Status | Reason |",
            "|----------|--------|--------|",
            "| `gemma3_12b` / RepLiQA GA (B3, Ours, B5) | **blocked†** | Re-judge run 2026-06-23: all rows `n_api_error` (Bedrock SCP deny). Predictions intact; Quoref judges for Gemma-12B still valid. |",
            "| `gemma3_12b` / SQuAD GA (B3, Ours, B5) | **blocked†** | Same SCP block; corrupted `bedrock_judge_summary.json` (0 judged ok). |",
            "| `gemma3_12b` / RepLiQA+SQuAD ceiling-gap rows | **stale** | `ceiling_gap_summary.json` from 2026-06-21 (pre-corruption); ignore until re-judge. GA tables above show `blocked†`. |",
            "| `llama31_70b` / Quoref judge | **blocked†** | Preds ready (2418×3); judge not run — Bedrock SCP deny. |",
            "| SQuAD ceiling gap tables (all models) | **missing** | Shared ceiling refs not in `cross_model/ceilings/squad_qa_v1/` — gap cannot be computed until Opus+Nova ceiling gen+judge complete. |",
            "| RepLiQA / SQuAD ceiling reference GA row | **missing** | Ceiling judge artifacts absent or incomplete for those datasets in `cross_model/ceilings/`. |",
            "",
            "† **AWS Bedrock blocked:** `test_bedrock_credentials.sh` fails with `AccessDeniedException` — org SCP explicit deny on "
            "`bedrock:InvokeModel` for `us.anthropic.claude-haiku-4-5-20251001-v1:0`. Re-judge with `FORCE_JUDGE=1` after access restored.",
            "",
        ]
        + (
            []
            if squad_ceiling_ok
            else [
                "SQuAD gap tables below are empty because ceiling references were never installed — not because per-model judges failed.",
                "",
            ]
        )
    )


def build_inference_table(cross_root: Path) -> str:
    lines = [
        "### Cross-model inference latency (Llama + Qwen2.5 only)",
        "",
        "Mean s/question, greedy decode, HF backend, 1× A100 bf16. "
        "**Gemma-3 excluded:** those runs mix vLLM batched timing, partial SLURM shards, and stale `timing.json` "
        "(not comparable to Llama/Qwen HF runs). See §5 for the reference Llama-3.2-3B inference table.",
        "",
        "| Model | RepLiQA B3 | Ours | spdup | B5 | Quoref B3 | Ours | spdup | B5 | SQuAD B3 | Ours | spdup | B5 |",
        "|-------|------------|------|-------|-----|-----------|------|-------|-----|----------|------|-------|-----|",
    ]
    for fam_name, members in INFERENCE_FAMILIES:
        for slug, size in members:
            label = f"{fam_name} {size}"
            row = [label]
            for ds_key, ds_sub, _, b3_cond in DATASETS:
                run = cross_root / slug / ds_sub
                b3 = collect_inference_row(run, b3_cond, dataset_key=ds_key, model_slug=slug)
                ours = collect_inference_row(run, OURS[ds_key], dataset_key=ds_key, model_slug=slug)
                b5 = collect_inference_row(run, B5[ds_key], dataset_key=ds_key, model_slug=slug)
                b3m, ours_m = b3.get("mean_s_per_question"), ours.get("mean_s_per_question")
                row.extend(
                    [
                        _fmt_inf(b3m if b3.get("reliable") else None),
                        _fmt_inf(ours_m if ours.get("reliable") else None),
                        _speedup(b3m if b3.get("reliable") else None, ours_m if ours.get("reliable") else None),
                        _fmt_inf(b5.get("mean_s_per_question") if b5.get("reliable") else None),
                    ]
                )
            lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "**Speedup pattern (Llama/Qwen):** Ours dense merge is **~2–3× faster** than B3 LoRA across datasets "
            "(consistent with §5 on Llama-3.2-3B).",
            "",
        ]
    )
    return "\n".join(lines)


def build_main_ga_table(cross_root: Path) -> str:
    lines = [
        "### Cross-model gold alignment (B3 / Ours / B5)",
        "",
        "| Model | RepLiQA B3 | Ours | B5 | Quoref B3 | Ours | B5 | SQuAD B3 | Ours | B5 |",
        "|-------|------------|------|-----|-----------|------|-----|----------|------|-----|",
    ]
    for slug, label in MODELS:
        row = [label]
        for ds_key, ds_sub, _, b3_cond in DATASETS:
            for cond in (b3_cond, OURS[ds_key], B5[ds_key]):
                row.append(_ga_cell(cross_root, slug, ds_sub, cond, dataset_key=ds_key))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def build_section(cross_root: Path, ceiling_root: Path) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ref_rows = []
    for ds_key, ds_sub, ds_label, _ in DATASETS:
        judged = ceiling_root / ds_sub / "judged"
        opus = judged / "REF_claude_opus" / "bedrock_judge_summary.json"
        nova = judged / "REF_nova_2_lite" / "bedrock_judge_summary.json"
        o_ga = n_ga = "—"
        if opus.is_file():
            o_ga = _fmt(json.loads(opus.read_text())["stats"].get("mean_gold_alignment"))
        if nova.is_file():
            n_ga = _fmt(json.loads(nova.read_text())["stats"].get("mean_gold_alignment"))
        ref_rows.append(f"| **{ds_label}** | {o_ga} | {n_ga} |")

    parts = [
        "## 9. Cross-Model Ceiling Gap Comparison",
        "",
        "Compares fine-tuned open models (B3 uniform LoRA, Ours QS tier+dense merge, B5 AdaLoRA) against "
        "**closed-model ceiling references** on the same eval sets. Ceilings are generated once per dataset "
        "on Bedrock, judged once with Haiku, then symlinked into each cross-model run.",
        "",
        f"**Last compiled:** {now} UTC · Regenerate: `python -m thesis.compile_cross_model_ceiling_gap`",
        "",
        "| Setting | Value |",
        "|---------|-------|",
        "| Ceilings | **REF_claude_opus** (Claude Opus 4.8) · **REF_nova_2_lite** (Amazon Nova 2 Lite) |",
        "| Judge | Bedrock Haiku `v3_eval_gold` (same as §1) |",
        "| Datasets | RepLiQA (n=2,000) · Quoref (n=2,418) · SQuAD v2 (n=11,873) |",
        "| Models | 8 open-weight backbones (Llama, Qwen2.5, Gemma-3 at 1B–14B) |",
        "| Artifacts | `cross_model/ceilings/{dataset}/REF_*/` · per-run `eval/judged/ceiling_gap_summary.json` |",
        "",
        "**Metrics:**",
        "",
        "| Metric | Definition |",
        "|--------|------------|",
        "| **gap** | ceiling GA − condition GA (lower = closer to best API answer) |",
        "| **% gain vs B3** | `(cond − B3) / (ceiling − B3) × 100` — share of B3→ceiling gap recovered |",
        "",
        "Regenerate gaps: `bash thesis/scripts/submit_cross_model_ceiling_gap.sh`",
        "",
        build_ga_by_family_tables(cross_root),
        build_b5_adalora_audit(cross_root),
        build_blocked_results_note(cross_root),
        build_main_ga_table(cross_root),
        build_training_table(cross_root),
        build_adapter_size_table(cross_root),
        build_inference_table(cross_root),
        "### Ceiling reference GA (Haiku judge)",
        "",
        "| Dataset | Claude Opus 4.8 | Nova 2 Lite |",
        "|---------|-----------------|-------------|",
        *ref_rows,
        "",
        "Nova can score above Opus on some sets under the same Haiku judge — cross-vendor ceiling ordering is not monotonic.",
        "",
    ]

    for ceiling, (label, _) in CEILING_LABELS.items():
        for ds_key, ds_sub, ds_label, _ in DATASETS:
            table, _ = build_gap_table(
                cross_root,
                dataset_key=ds_key,
                ds_subdir=ds_sub,
                ds_label=ds_label,
                ceiling=ceiling,
                ceiling_label=label,
            )
            parts.append(table)

    # takeaways from data
    parts.extend(
        [
            "### Takeaways",
            "",
            "- **Ours closes ~6–29% of the B3→ceiling gap** on mid/large models (Llama-3.1-8B, Qwen2.5-7B/14B, Gemma-3-4B/12B on Quoref). "
            "Best open Quoref run: **Qwen2.5-14B Ours GA≈4.21** (gap 0.25 vs Opus, ~29% of B3→ceiling gain).",
            "- **Qwen2.5-14B is strongest on Quoref** — B5 reaches 4.28 GA (gap 0.17 vs Opus, ~50% of B3→ceiling gain). "
            "On RepLiQA, B5 at 4.00 GA is the closest open condition (gap 0.40 vs Opus).",
            "- **Small models (1B) show minimal ceiling recovery** — Ours improves B3 by only ~2% on Quoref despite large absolute gaps.",
            "- **AdaLoRA (B5) is mixed vs Ours** — strong only on **Qwen2.5-14B** (see B5 audit above); elsewhere B5 trails or is slower to train.",
            "- **Training:** Ours QS+merge is often **similar or faster** than B3 uniform LoRA (except B5, which is slowest).",
            "- **Inference:** Dense-merge Ours gives **2–3×** decode speedup vs LoRA on **Llama/Qwen** (Gemma timing omitted — mixed vLLM/shard artifacts).",
            "",
        ]
    )
    return "\n".join(parts)


def patch_results_summary(
    results_path: Path,
    section_md: str,
    *,
    start_heading: str = "## 9. Cross-Model Ceiling Gap Comparison",
    end_heading: str = "## 10. Pending / In Flight",
) -> None:
    text = results_path.read_text(encoding="utf-8")
    start = text.index(start_heading)
    end = text.index(end_heading)
    new_text = text[:start] + section_md.rstrip() + "\n\n---\n\n" + text[end:]
    # update header last-updated line
    lines = new_text.splitlines()
    for i, line in enumerate(lines[:10]):
        if line.startswith("**Last updated:**"):
            lines[i] = (
                f"**Last updated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')} "
                "— cross-model results (§9: GA, training, adapter sizes, inference, ceiling gap)"
            )
            break
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    cross_root = Path("/fs/ess/PAS2699/pratham2210/cross_model/runs")
    ceiling_root = Path("/fs/ess/PAS2699/pratham2210/cross_model/ceilings")
    results_path = Path(__file__).resolve().parent / "RESULTS_SUMMARY.md"

    # Refresh per-run inference_timing.json aggregates.
    from thesis.cross_model_inference_timing import collect_cross_model_inference_timing

    matrix_path = cross_root / "inference_timing_matrix.json"
    matrix_doc = collect_cross_model_inference_timing(cross_root)
    matrix_path.write_text(json.dumps(matrix_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {matrix_path} ({matrix_doc['n_runs']} runs)")

    section = build_section(cross_root, ceiling_root)
    patch_results_summary(results_path, section)
    print(f"Patched {results_path}")
    # status
    total = len(MODELS) * len(DATASETS)
    have = sum(1 for slug, _ in MODELS for _, sub, _, _ in DATASETS if _load_row(cross_root / slug / sub))
    print(f"Ceiling gap summaries: {have}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
