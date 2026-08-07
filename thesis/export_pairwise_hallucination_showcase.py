"""
Pairwise hallucination showcases: Ours vs B3 and Ours vs B5 (separate files).

  python -m thesis.cli export-pairwise-hallucination-showcase
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from thesis.eval_export_hallucination_pack import (
    _classify_refusal,
    _is_invented,
    _is_refusal_like,
    _judge_block,
    _load_jsonl_map,
    find_judge_gap_cases,
    find_refusal_vs_invent_cases,
)
from thesis.export_triple_hallucination_catalog import (
    CROSS_DATASET_DIRS,
    DATASET_CONDITIONS,
    DEFAULT_CROSS_ROOT,
    MODEL_META,
    REFERENCE_RUNS,
)
from thesis.qa_answer_metrics import is_refusal_gold

THESIS_ROOT = Path(__file__).resolve().parent
Baseline = Literal["B3", "B5"]
SCHEMA = "pairwise_hallucination_showcase/v1"

MODEL_ORDER = [
    "Llama-3.2-1B",
    "Llama-3.2-3B",
    "Llama-3.1-8B",
    "Llama-3.1-70B",
    "Qwen2.5-3B",
    "Qwen2.5-7B",
    "Qwen2.5-14B",
    "Gemma-3-1B",
    "Gemma-3-4B",
    "Gemma-3-12B",
]


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str | None, limit: int = 500) -> str | None:
    if text is None:
        return None
    t = str(text).strip()
    return t if len(t) <= limit else t[: limit - 3] + "..."



def find_pairwise_refusal_cases(
    *,
    baseline_rows: dict[str, dict[str, Any]],
    ours_rows: dict[str, dict[str, Any]],
    baseline: Baseline,
) -> list[dict[str, Any]]:
    raw = find_refusal_vs_invent_cases(
        b3_rows=baseline_rows, ours_rows=ours_rows, require_judge=True
    )
    out: list[dict[str, Any]] = []
    for row in raw:
        out.append(
            {
                "eval_id": row["eval_id"],
                "pattern": "refusal",
                "question": row["question"],
                "gold": row["gold"],
                "context": None,
                baseline: {
                    "pred": row["b3_pred"],
                    "gold_alignment": row["b3_gold_alignment"],
                    "grounding": row["b3_grounding"],
                    "refusal_class": row["b3_refusal_class"],
                    "judge_reason": row.get("b3_judge_reason"),
                },
                "Ours": {
                    "pred": row["ours_pred"],
                    "gold_alignment": row["ours_gold_alignment"],
                    "grounding": row["ours_grounding"],
                    "refusal_class": row["ours_refusal_class"],
                    "judge_reason": row.get("ours_judge_reason"),
                },
                "_score": float(row["ours_gold_alignment"] or 0)
                - float(row["b3_gold_alignment"] or 0),
            }
        )
    out.sort(key=lambda r: (-r["_score"], r["eval_id"]))
    for r in out:
        r.pop("_score", None)
    return out


def find_pairwise_judge_gap_cases(
    *,
    baseline_rows: dict[str, dict[str, Any]],
    ours_rows: dict[str, dict[str, Any]],
    baseline: Baseline,
) -> list[dict[str, Any]]:
    raw = find_judge_gap_cases(
        b3_rows=baseline_rows, ours_rows=ours_rows, require_judge=True
    )
    out: list[dict[str, Any]] = []
    for row in raw:
        out.append(
            {
                "eval_id": row["eval_id"],
                "pattern": "judge_gap",
                "question": row["question"],
                "gold": row["gold"],
                "context": None,
                baseline: {
                    "pred": row["b3_pred"],
                    "gold_alignment": row["b3_gold_alignment"],
                    "grounding": row["b3_grounding"],
                    "refusal_class": row["b3_refusal_class"],
                    "judge_reason": row.get("b3_judge_reason"),
                },
                "Ours": {
                    "pred": row["ours_pred"],
                    "gold_alignment": row["ours_gold_alignment"],
                    "grounding": row["ours_grounding"],
                    "refusal_class": row["ours_refusal_class"],
                    "judge_reason": row.get("ours_judge_reason"),
                },
                "_score": float(row["ours_gold_alignment"] or 0)
                - float(row["b3_gold_alignment"] or 0),
            }
        )
    out.sort(key=lambda r: (-r["_score"], r["eval_id"]))
    for r in out:
        r.pop("_score", None)
    return out


def _enrich_context(cases: list[dict[str, Any]], ours_rows: dict[str, dict[str, Any]]) -> None:
    for c in cases:
        base = ours_rows.get(c["eval_id"], {})
        c["context"] = base.get("context")
        if not c.get("question"):
            c["question"] = base.get("question")


def _collect_run(
    *,
    model_slug: str,
    dataset: str,
    run_root: Path,
    source: str,
    baseline: Baseline,
) -> list[dict[str, Any]]:
    b3_cond, b5_cond, ours_cond = DATASET_CONDITIONS[dataset]
    bl_cond = b3_cond if baseline == "B3" else b5_cond
    judged = run_root / "eval" / "judged"
    bl_path = judged / bl_cond / "bedrock_judge.jsonl"
    ours_path = judged / ours_cond / "bedrock_judge.jsonl"
    if not bl_path.is_file() or not ours_path.is_file():
        return []

    bl_rows = _load_jsonl_map(bl_path)
    ours_rows = _load_jsonl_map(ours_path)
    meta = MODEL_META.get(model_slug, {"label": model_slug})

    refusal = find_pairwise_refusal_cases(
        baseline_rows=bl_rows, ours_rows=ours_rows, baseline=baseline
    )
    gap = find_pairwise_judge_gap_cases(
        baseline_rows=bl_rows, ours_rows=ours_rows, baseline=baseline
    )
    cases: list[dict[str, Any]] = []
    for row in refusal + gap:
        _enrich_context([row], ours_rows)
        cases.append(
            {
                "case_id": f"{model_slug}/{dataset}/{baseline}/{row['pattern']}/{row['eval_id']}",
                "eval_id": row["eval_id"],
                "dataset": dataset,
                "pattern": row["pattern"],
                "baseline": baseline,
                "model_slug": model_slug,
                "model_label": meta["label"],
                "model_family": meta.get("family", ""),
                "model_size": meta.get("size", ""),
                "source": source,
                "run_root": str(run_root),
                "baseline_condition": bl_cond,
                "ours_condition": ours_cond,
                "question": row.get("question"),
                "gold": row.get("gold"),
                "context": row.get("context"),
                baseline: row[baseline],
                "Ours": row["Ours"],
            }
        )
    return cases


def build_pairwise_catalog(
    baseline: Baseline,
    *,
    cross_root: Path = DEFAULT_CROSS_ROOT,
) -> dict[str, Any]:
    all_cases: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []

    def add_cases(model_slug: str, dataset: str, run_root: Path, source: str) -> None:
        cases = _collect_run(
            model_slug=model_slug,
            dataset=dataset,
            run_root=run_root,
            source=source,
            baseline=baseline,
        )
        if not cases:
            return
        refusal_n = sum(1 for c in cases if c["pattern"] == "refusal")
        gap_n = sum(1 for c in cases if c["pattern"] == "judge_gap")
        per_run.append(
            {
                "model_slug": model_slug,
                "model_label": MODEL_META.get(model_slug, {}).get("label", model_slug),
                "dataset": dataset,
                "source": source,
                "n_refusal": refusal_n,
                "n_judge_gap": gap_n,
                "n_total": len(cases),
            }
        )
        all_cases.extend(cases)

    for model_slug, dataset, run_root, source in REFERENCE_RUNS:
        add_cases(model_slug, dataset, run_root, source)

    for model_slug in MODEL_META:
        if model_slug == "llama32_3b":
            continue
        for dataset, ds_dir in CROSS_DATASET_DIRS.items():
            run_root = cross_root / model_slug / ds_dir
            if run_root.is_dir():
                add_cases(model_slug, dataset, run_root, "cross_model")

    by_model: dict[str, int] = defaultdict(int)
    by_pattern: dict[str, int] = defaultdict(int)
    by_dataset: dict[str, int] = defaultdict(int)
    for c in all_cases:
        by_model[c["model_label"]] += 1
        by_pattern[c["pattern"]] += 1
        by_dataset[c["dataset"]] += 1

    return {
        "schema": f"pairwise_hallucination_catalog/v1",
        "baseline": baseline,
        "created_at": utc_iso(),
        "description": (
            f"Cases where Ours is correct (GA≥4 or proper refusal) but {baseline} fails."
        ),
        "summary": {
            "n_cases_total": len(all_cases),
            "n_runs": len(per_run),
            "by_model_label": dict(sorted(by_model.items())),
            "by_pattern": dict(sorted(by_pattern.items())),
            "by_dataset": dict(sorted(by_dataset.items())),
            "per_run": per_run,
        },
        "cases": all_cases,
    }


def _score_pairwise(case: dict[str, Any], baseline: Baseline) -> float:
    bl = case[baseline]
    return float(case["Ours"].get("gold_alignment") or 0) - float(
        bl.get("gold_alignment") or 0
    )


def _pick_examples(
    cases: list[dict[str, Any]],
    baseline: Baseline,
    *,
    n_total: int = 20,
) -> list[dict[str, Any]]:
    strong = [c for c in cases if float(c["Ours"].get("gold_alignment") or 0) >= 4.0]
    if not strong:
        return []

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for c in strong:
        buckets[(c["pattern"], c["dataset"])].append(c)
    for lst in buckets.values():
        lst.sort(key=lambda c: _score_pairwise(c, baseline), reverse=True)

    order = [
        ("refusal", "repliqa"),
        ("refusal", "squad"),
        ("refusal", "quoref"),
        ("judge_gap", "repliqa"),
        ("judge_gap", "quoref"),
        ("judge_gap", "squad"),
    ]
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(picked) < n_total:
        added = False
        for key in order:
            if len(picked) >= n_total:
                break
            for c in buckets.get(key, []):
                if c["case_id"] in seen:
                    continue
                seen.add(c["case_id"])
                picked.append(c)
                added = True
                break
        if not added:
            break
    if len(picked) < n_total:
        for c in sorted(strong, key=lambda x: _score_pairwise(x, baseline), reverse=True):
            if c["case_id"] in seen:
                continue
            picked.append(c)
            if len(picked) >= n_total:
                break
    picked.sort(
        key=lambda c: (
            0 if c["pattern"] == "refusal" else 1,
            c["dataset"],
            -_score_pairwise(c, baseline),
        )
    )
    return picked[:n_total]


def _format_case(case: dict[str, Any], baseline: Baseline) -> dict[str, Any]:
    bl = dict(case[baseline])
    ours = dict(case["Ours"])
    bl["pred"] = _truncate(bl.get("pred"), 500)
    ours["pred"] = _truncate(ours.get("pred"), 500)
    bl["judge_reason"] = _truncate(bl.get("judge_reason"), 300)
    ours["judge_reason"] = _truncate(ours.get("judge_reason"), 300)
    return {
        **case,
        "context": _truncate(case.get("context"), 400),
        "score_gap": round(_score_pairwise(case, baseline), 2),
        baseline: bl,
        "Ours": ours,
    }


def _compute_pairwise_stats(catalog: dict[str, Any], baseline: Baseline) -> dict[str, Any]:
    """Per-model counts: Ours wins vs baseline only (not requiring other baseline to fail)."""
    from thesis.export_triple_hallucination_showcase import _compute_hallucination_stats

    cross_root = DEFAULT_CROSS_ROOT
    stats = _compute_hallucination_stats(cross_root)
    bl_key = baseline

    by_model_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in catalog["cases"]:
        by_model_cases[c["model_label"]].append(c)

    per_model: list[dict[str, Any]] = []
    for ms in stats["per_model"]:
        label = ms["model_label"]
        cases = by_model_cases.get(label, [])
        refusal_n = sum(1 for c in cases if c["pattern"] == "refusal")
        gap_n = sum(1 for c in cases if c["pattern"] == "judge_gap")
        strong = sum(
            1 for c in cases if float(c["Ours"].get("gold_alignment") or 0) >= 4
        )
        hr_bl = ms["hallucination_rate"][bl_key]
        hr_ours = ms["hallucination_rate"]["Ours"]
        per_model.append(
            {
                "model_label": label,
                "model_slug": ms["model_slug"],
                "n_eval_rows": ms["n_eval_rows"],
                f"{bl_key}_hallucination_rate": hr_bl,
                "Ours_hallucination_rate": hr_ours,
                "hallucination_rate_delta_ours_vs_baseline": round(hr_bl - hr_ours, 4),
                "pairwise_proof_n": len(cases),
                "pairwise_refusal_n": refusal_n,
                "pairwise_judge_gap_n": gap_n,
                "pairwise_strong_n": strong,
                "by_dataset": ms.get("by_dataset", {}),
            }
        )

    return {
        "baseline": baseline,
        "per_model": per_model,
        "catalog_summary": catalog["summary"],
    }


def build_showcase(
    catalog: dict[str, Any],
    baseline: Baseline,
    *,
    examples_per_model: int = 20,
) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in catalog["cases"]:
        by_model[case["model_label"]].append(case)

    models_out: list[dict[str, Any]] = []
    order_idx = {m: i for i, m in enumerate(MODEL_ORDER)}
    for label in sorted(by_model.keys(), key=lambda x: order_idx.get(x, 999)):
        cases = by_model[label]
        examples = _pick_examples(cases, baseline, n_total=examples_per_model)
        models_out.append(
            {
                "model_label": label,
                "model_slug": examples[0]["model_slug"] if examples else None,
                "n_available": len(cases),
                "n_showcase": len(examples),
                "n_refusal_examples": sum(1 for e in examples if e["pattern"] == "refusal"),
                "n_judge_gap_examples": sum(1 for e in examples if e["pattern"] == "judge_gap"),
                "examples": [_format_case(e, baseline) for e in examples],
            }
        )

    # Highlights
    all_strong = [
        c for c in catalog["cases"] if float(c["Ours"].get("gold_alignment") or 0) >= 4
    ]
    refusals = sorted(
        [c for c in all_strong if c["pattern"] == "refusal"],
        key=lambda c: _score_pairwise(c, baseline),
        reverse=True,
    )
    gaps = sorted(
        [c for c in all_strong if c["pattern"] == "judge_gap"],
        key=lambda c: _score_pairwise(c, baseline),
        reverse=True,
    )
    highlights: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pool, limit in ((refusals, 8), (gaps, 8)):
        for c in pool:
            if c["case_id"] in seen:
                continue
            if sum(1 for h in highlights if h["model_label"] == c["model_label"]) >= 3:
                continue
            seen.add(c["case_id"])
            highlights.append(_format_case(c, baseline))
            if len([h for h in highlights if h["pattern"] == c["pattern"]]) >= limit:
                break

    return {
        "schema": SCHEMA,
        "baseline": baseline,
        "comparison": f"Ours vs {baseline}",
        "created_at": utc_iso(),
        "examples_per_model": examples_per_model,
        "statistics": _compute_pairwise_stats(catalog, baseline),
        "catalog_summary": catalog["summary"],
        "highlights": highlights,
        "models": models_out,
    }


def _write_markdown(showcase: dict[str, Any], path: Path) -> None:
    baseline: Baseline = showcase["baseline"]
    bl = baseline
    stats = showcase["statistics"]["per_model"]
    stats_by = {s["model_label"]: s for s in stats}

    lines = [
        f"# Ours vs {baseline} — Hallucination Showcase",
        "",
        f"Cases where **Ours is correct** but **{baseline}** hallucinates or invents.",
        "",
        f"**Generated:** {showcase['created_at']}  ",
        f"**Examples per model:** {showcase['examples_per_model']}  ",
        f"**Catalog:** `ours_vs_{bl.lower()}_catalog.json` ({showcase['catalog_summary']['n_cases_total']} cases)",
        "",
        "---",
        "",
        "## Summary statistics",
        "",
        f"| Model | Eval rows | Halluc. {baseline} | Halluc. Ours | Δ Ours−{baseline} | "
        f"Pairwise proofs | Refusal | Answer |",
        "|-------|-----------|----------------|--------------|----------------|-----------------|---------|--------|",
    ]
    for label in MODEL_ORDER:
        s = stats_by.get(label)
        if not s:
            continue
        lines.append(
            f"| {label} | {s['n_eval_rows']} | "
            f"{s[f'{bl}_hallucination_rate']:.1%} | {s['Ours_hallucination_rate']:.1%} | "
            f"{s['hallucination_rate_delta_ours_vs_baseline']:+.1%} | "
            f"{s['pairwise_proof_n']} | {s['pairwise_refusal_n']} | {s['pairwise_judge_gap_n']} |"
        )

    lines.extend(
        [
            "",
            "Positive Δ = Ours has **lower** hallucination rate than "
            f"{baseline}.",
            "",
            "### RepLiQA hallucination rate",
            "",
            f"| Model | {baseline} | Ours | Δ |",
            "|-------|------|------|---|",
        ]
    )
    for label in MODEL_ORDER:
        s = stats_by.get(label)
        if not s or "repliqa" not in s.get("by_dataset", {}):
            continue
        d = s["by_dataset"]["repliqa"]
        hr = d["hallucination_rate"]
        lines.append(
            f"| {label} | {hr[bl]:.1%} | {hr['Ours']:.1%} | "
            f"{d['hallucination_rate_delta_ours_vs_b3' if bl=='B3' else 'hallucination_rate_delta_ours_vs_b5']:+.1%} |"
        )

    lines.extend(["", "---", "", "## Highlights", ""])
    for i, ex in enumerate(showcase.get("highlights", []), start=1):
        _write_example(lines, ex, i, baseline)

    lines.extend(["", "---", "", "## Per-model examples", ""])
    for block in showcase["models"]:
        lines.extend(
            [
                f"## {block['model_label']} "
                f"({block['n_showcase']} examples · "
                f"{block['n_refusal_examples']} refusal / "
                f"{block['n_judge_gap_examples']} answer · "
                f"{block['n_available']} in catalog)",
                "",
            ]
        )
        for i, ex in enumerate(block["examples"], start=1):
            _write_example(lines, ex, i, baseline)

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_example(
    lines: list[str], ex: dict[str, Any], i: int, baseline: Baseline
) -> None:
    pat = "Refusal win" if ex["pattern"] == "refusal" else "Answer win"
    bl = ex[baseline]
    lines.extend(
        [
            f"### {i}. `{ex['eval_id']}` — {pat} ({ex['dataset']})",
            "",
            f"**Q:** {ex['question']}",
            "",
            f"**Gold:** {ex['gold']}",
            "",
            f"**{baseline}** (GA={bl['gold_alignment']}, G={bl['grounding']}):",
            "",
            str(bl["pred"]),
            "",
            f"**Ours** (GA={ex['Ours']['gold_alignment']}, G={ex['Ours']['grounding']}):",
            "",
            str(ex["Ours"]["pred"]),
            "",
        ]
    )
    if ex["Ours"].get("judge_reason"):
        lines.append(f"*Judge (Ours):* {ex['Ours']['judge_reason']}")
        lines.append("")
    lines.append("---")
    lines.append("")


def run_export_pairwise_hallucination_showcase(ns: argparse.Namespace) -> int:
    out_dir = Path(ns.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cross_root = Path(ns.cross_root).expanduser().resolve()
    n = int(ns.examples_per_model)

    for baseline in ("B3", "B5"):
        print(f"Building Ours vs {baseline} catalog...", flush=True)
        catalog = build_pairwise_catalog(baseline, cross_root=cross_root)
        bl = baseline.lower()
        cat_path = out_dir / f"ours_vs_{bl}_catalog.json"
        cat_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Wrote {cat_path} ({catalog['summary']['n_cases_total']} cases)")

        showcase = build_showcase(catalog, baseline, examples_per_model=n)
        json_path = out_dir / f"ours_vs_{bl}_showcase.json"
        md_path = out_dir / f"ours_vs_{bl}_showcase.md"
        stats_path = out_dir / f"ours_vs_{bl}_stats.json"

        json_path.write_text(json.dumps(showcase, indent=2, ensure_ascii=False), encoding="utf-8")
        stats_path.write_text(
            json.dumps(
                {
                    "schema": "pairwise_hallucination_stats/v1",
                    "created_at": showcase["created_at"],
                    **showcase["statistics"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _write_markdown(showcase, md_path)
        print(f"  Wrote {md_path}")
        print(f"  Wrote {json_path}")
        print(f"  Wrote {stats_path}")

    return 0


def add_cli(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "export-pairwise-hallucination-showcase",
        help="Separate Ours vs B3 and Ours vs B5 showcase files per model",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=THESIS_ROOT / "experiments/analysis",
    )
    p.add_argument("--cross-root", type=Path, default=DEFAULT_CROSS_ROOT)
    p.add_argument("--examples-per-model", type=int, default=20)
    p.set_defaults(fn=run_export_pairwise_hallucination_showcase)
