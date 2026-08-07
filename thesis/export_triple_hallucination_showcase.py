"""
Curated showcase (~10 examples per model) + hallucination statistics from triple catalog.

  python -m thesis.cli export-triple-hallucination-showcase
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.eval_export_hallucination_pack import (
    _classify_refusal,
    _is_invented,
    _is_refusal_like,
    _judge_block,
    _load_jsonl_map,
)
from thesis.export_triple_hallucination_catalog import (
    CROSS_DATASET_DIRS,
    DATASET_CONDITIONS,
    DEFAULT_CROSS_ROOT,
    MODEL_META,
    REFERENCE_RUNS,
    build_catalog,
)
from thesis.qa_answer_metrics import is_refusal_gold

THESIS_ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = THESIS_ROOT / "experiments/analysis/triple_hallucination_catalog.json"
SCHEMA = "triple_hallucination_showcase/v1"

DATASET_EVAL_N = {"repliqa": 2000, "quoref": 2418, "squad": 11873}


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_hallucination_row(row: dict[str, Any]) -> bool:
    gold = str(row.get("gold") or "")
    pred = str(row.get("pred") or "")
    j = _judge_block(row)
    ga = float(j.get("gold_alignment") or 0)
    if is_refusal_gold(gold):
        return _is_invented(pred) or ga <= 2
    return ga <= 2


def _is_correct_refusal_row(row: dict[str, Any]) -> bool:
    gold = str(row.get("gold") or "")
    pred = str(row.get("pred") or "")
    if not is_refusal_gold(gold):
        return False
    j = _judge_block(row)
    ga = float(j.get("gold_alignment") or 0)
    return _is_refusal_like(pred) and ga >= 4


def _score_case(case: dict[str, Any]) -> float:
    oga = float(case["Ours"].get("gold_alignment") or 0)
    b3ga = float(case["B3"].get("gold_alignment") or 0)
    b5ga = float(case["B5"].get("gold_alignment") or 0)
    return min(oga - b3ga, oga - b5ga)


def _truncate(text: str | None, limit: int = 600) -> str | None:
    if text is None:
        return None
    t = str(text).strip()
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


def _pick_examples(
    cases: list[dict[str, Any]],
    *,
    n_total: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if not cases:
        return []
    # Prefer strong Ours wins for thesis display
    cases = [
        c
        for c in cases
        if float(c["Ours"].get("gold_alignment") or 0) >= 4.0
    ]
    if not cases:
        return []

    # Bucket by (pattern, dataset) for diversity
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        buckets[(c["pattern"], c["dataset"])].append(c)
    for lst in buckets.values():
        lst.sort(key=_score_case, reverse=True)

    # Prefer refusal first, then answer wins; cycle datasets
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
    # Round-robin from buckets so we get mix of refusal + answer across datasets
    while len(picked) < n_total:
        added = False
        for key in order:
            if len(picked) >= n_total:
                break
            pool = buckets.get(key, [])
            for c in pool:
                cid = c["case_id"]
                if cid in seen:
                    continue
                seen.add(cid)
                picked.append(c)
                added = True
                break
        if not added:
            break

    remaining = n_total - len(picked)
    if remaining > 0:
        for c in sorted(cases, key=_score_case, reverse=True):
            if c["case_id"] in seen:
                continue
            picked.append(c)
            seen.add(c["case_id"])
            if len(picked) >= n_total:
                break

    picked.sort(key=lambda c: (0 if c["pattern"] == "refusal" else 1, c["dataset"], -_score_case(c)))
    return picked[:n_total]


def _compute_hallucination_stats(cross_root: Path) -> dict[str, Any]:
    per_model: dict[str, dict[str, Any]] = {}
    per_run: list[dict[str, Any]] = []

    def process_run(model_slug: str, dataset: str, run_root: Path, source: str) -> None:
        b3_cond, b5_cond, ours_cond = DATASET_CONDITIONS[dataset]
        judged = run_root / "eval" / "judged"
        paths = {
            "B3": judged / b3_cond / "bedrock_judge.jsonl",
            "B5": judged / b5_cond / "bedrock_judge.jsonl",
            "Ours": judged / ours_cond / "bedrock_judge.jsonl",
        }
        if not all(p.is_file() for p in paths.values()):
            return
        rows = {k: _load_jsonl_map(p) for k, p in paths.items()}
        n = max(len(v) for v in rows.values())
        label = MODEL_META[model_slug]["label"]

        run_stat: dict[str, Any] = {
            "model_slug": model_slug,
            "model_label": label,
            "dataset": dataset,
            "source": source,
            "n_eval": n,
            "hallucination_rate": {},
            "refusal_correct_rate": {},
            "unanswerable_invent_rate": {},
            "answerable_wrong_rate": {},
        }
        for cond, data in rows.items():
            hall = sum(1 for r in data.values() if _is_hallucination_row(r))
            run_stat["hallucination_rate"][cond] = round(hall / max(n, 1), 4)

            unans = [r for r in data.values() if is_refusal_gold(str(r.get("gold") or ""))]
            if unans:
                invent = sum(1 for r in unans if _is_invented(str(r.get("pred") or "")))
                run_stat["unanswerable_invent_rate"][cond] = round(invent / len(unans), 4)
                correct_ref = sum(1 for r in unans if _is_correct_refusal_row(r))
                run_stat["refusal_correct_rate"][cond] = round(correct_ref / len(unans), 4)

            ans = [r for r in data.values() if not is_refusal_gold(str(r.get("gold") or ""))]
            if ans:
                wrong = sum(
                    1
                    for r in ans
                    if float(_judge_block(r).get("gold_alignment") or 0) <= 2
                )
                run_stat["answerable_wrong_rate"][cond] = round(wrong / len(ans), 4)

        per_run.append(run_stat)
        agg = per_model.setdefault(
            label,
            {
                "model_slug": model_slug,
                "n_eval_rows": 0,
                "hallucination_n": {"B3": 0, "B5": 0, "Ours": 0},
                "unanswerable_n": 0,
                "unanswerable_invent_n": {"B3": 0, "B5": 0, "Ours": 0},
                "refusal_correct_n": {"B3": 0, "B5": 0, "Ours": 0},
                "answerable_n": 0,
                "answerable_wrong_n": {"B3": 0, "B5": 0, "Ours": 0},
                "triple_proof_n": 0,
                "by_dataset": {},
            },
        )
        agg["n_eval_rows"] += n
        # Count unanswerable/answerable once (from Ours gold labels)
        gold_rows = list(rows["Ours"].values())
        unans_ids = {
            str(r.get("eval_id") or r.get("chunk_id") or "")
            for r in gold_rows
            if is_refusal_gold(str(r.get("gold") or ""))
        }
        ans_ids = {
            str(r.get("eval_id") or r.get("chunk_id") or "")
            for r in gold_rows
            if not is_refusal_gold(str(r.get("gold") or ""))
        }
        agg["unanswerable_n"] += len(unans_ids)
        agg["answerable_n"] += len(ans_ids)

        ds_agg = agg["by_dataset"].setdefault(
            dataset,
            {
                "n_eval": 0,
                "hallucination_n": {"B3": 0, "B5": 0, "Ours": 0},
                "unanswerable_n": 0,
                "unanswerable_invent_n": {"B3": 0, "B5": 0, "Ours": 0},
                "refusal_correct_n": {"B3": 0, "B5": 0, "Ours": 0},
                "answerable_n": 0,
                "answerable_wrong_n": {"B3": 0, "B5": 0, "Ours": 0},
            },
        )
        ds_agg["n_eval"] += n
        ds_agg["unanswerable_n"] += len(unans_ids)
        ds_agg["answerable_n"] += len(ans_ids)

        for cond, data in rows.items():
            hall_n = sum(1 for r in data.values() if _is_hallucination_row(r))
            invent_n = sum(
                1
                for r in data.values()
                if is_refusal_gold(str(r.get("gold") or ""))
                and _is_invented(str(r.get("pred") or ""))
            )
            refuse_n = sum(1 for r in data.values() if _is_correct_refusal_row(r))
            wrong_n = sum(
                1
                for r in data.values()
                if not is_refusal_gold(str(r.get("gold") or ""))
                and float(_judge_block(r).get("gold_alignment") or 0) <= 2
            )
            agg["hallucination_n"][cond] += hall_n
            agg["unanswerable_invent_n"][cond] += invent_n
            agg["refusal_correct_n"][cond] += refuse_n
            agg["answerable_wrong_n"][cond] += wrong_n
            ds_agg["hallucination_n"][cond] += hall_n
            ds_agg["unanswerable_invent_n"][cond] += invent_n
            ds_agg["refusal_correct_n"][cond] += refuse_n
            ds_agg["answerable_wrong_n"][cond] += wrong_n

    for model_slug, dataset, run_root, source in REFERENCE_RUNS:
        process_run(model_slug, dataset, run_root, source)
    for model_slug in MODEL_META:
        if model_slug == "llama32_3b":
            continue
        for dataset, ds_dir in CROSS_DATASET_DIRS.items():
            run_root = cross_root / model_slug / ds_dir
            if run_root.is_dir():
                process_run(model_slug, dataset, run_root, "cross_model")

    def _rates(agg: dict[str, Any], n_key: str = "n_eval_rows") -> dict[str, Any]:
        n = max(agg.get(n_key) or agg.get("n_eval") or 1, 1)
        un = agg.get("unanswerable_n") or 0
        an = agg.get("answerable_n") or 0
        return {
            "hallucination_rate": {
                k: round(agg["hallucination_n"][k] / n, 4) for k in ("B3", "B5", "Ours")
            },
            "unanswerable_invent_rate": {
                k: round(agg["unanswerable_invent_n"][k] / un, 4) if un else None
                for k in ("B3", "B5", "Ours")
            },
            "refusal_correct_rate": {
                k: round(agg["refusal_correct_n"][k] / un, 4) if un else None
                for k in ("B3", "B5", "Ours")
            },
            "answerable_wrong_rate": {
                k: round(agg["answerable_wrong_n"][k] / an, 4) if an else None
                for k in ("B3", "B5", "Ours")
            },
            "hallucination_rate_delta_ours_vs_b3": round(
                agg["hallucination_n"]["B3"] / n - agg["hallucination_n"]["Ours"] / n, 4
            ),
            "hallucination_rate_delta_ours_vs_b5": round(
                agg["hallucination_n"]["B5"] / n - agg["hallucination_n"]["Ours"] / n, 4
            ),
        }

    # finalize aggregated rates
    model_stats: list[dict[str, Any]] = []
    for label, agg in sorted(per_model.items()):
        rates = _rates(agg)
        by_ds = {}
        for ds, ds_agg in agg.get("by_dataset", {}).items():
            by_ds[ds] = {
                "n_eval": ds_agg["n_eval"],
                "unanswerable_n": ds_agg["unanswerable_n"],
                "answerable_n": ds_agg["answerable_n"],
                **_rates(ds_agg, "n_eval"),
            }
        model_stats.append(
            {
                "model_label": label,
                "model_slug": agg["model_slug"],
                "n_eval_rows": agg["n_eval_rows"],
                "unanswerable_n": agg["unanswerable_n"],
                "answerable_n": agg["answerable_n"],
                **rates,
                "by_dataset": by_ds,
            }
        )

    return {"per_run": per_run, "per_model": model_stats}


def _format_example(ex: dict[str, Any]) -> dict[str, Any]:
    out = {
        **ex,
        "context": _truncate(ex.get("context"), 400),
        "score_gap": round(_score_case(ex), 2),
    }
    for key in ("B3", "B5", "Ours"):
        if key in out and isinstance(out[key], dict):
            out[key] = dict(out[key])
            out[key]["pred"] = _truncate(out[key].get("pred"), 500)
            out[key]["judge_reason"] = _truncate(out[key].get("judge_reason"), 300)
    return out


def build_showcase(
    catalog: dict[str, Any],
    *,
    examples_per_model: int = 20,
    seed: int = 42,
    cross_root: Path = DEFAULT_CROSS_ROOT,
) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in catalog["cases"]:
        by_model[case["model_label"]].append(case)

    triple_counts = {label: len(cases) for label, cases in by_model.items()}
    triple_by_pattern: dict[str, dict[str, int]] = {}
    for label, cases in by_model.items():
        strong = [c for c in cases if float(c["Ours"].get("gold_alignment") or 0) >= 4]
        triple_by_pattern[label] = {
            "refusal": sum(1 for c in strong if c["pattern"] == "refusal"),
            "judge_gap": sum(1 for c in strong if c["pattern"] == "judge_gap"),
            "strong_total": len(strong),
            "all_total": len(cases),
        }

    stats = _compute_hallucination_stats(cross_root)
    for ms in stats["per_model"]:
        label = ms["model_label"]
        ms["triple_proof_n"] = triple_counts.get(label, 0)
        ms["triple_proof_rate"] = round(
            triple_counts.get(label, 0) / max(ms["n_eval_rows"], 1), 4
        )
        ms["triple_by_pattern"] = triple_by_pattern.get(label, {})

    model_order = [
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
    order_idx = {m: i for i, m in enumerate(model_order)}
    showcase_models: list[dict[str, Any]] = []
    all_strong: list[dict[str, Any]] = []
    for label in sorted(by_model.keys(), key=lambda x: order_idx.get(x, 999)):
        cases = by_model[label]
        examples = _pick_examples(cases, n_total=examples_per_model, seed=seed)
        showcase_cases = [_format_example(ex) for ex in examples]
        refusal_n = sum(1 for c in examples if c["pattern"] == "refusal")
        gap_n = sum(1 for c in examples if c["pattern"] == "judge_gap")
        showcase_models.append(
            {
                "model_label": label,
                "model_slug": examples[0]["model_slug"] if examples else None,
                "n_available": len(cases),
                "n_showcase": len(examples),
                "n_refusal_examples": refusal_n,
                "n_judge_gap_examples": gap_n,
                "examples": showcase_cases,
            }
        )
        for c in cases:
            if float(c["Ours"].get("gold_alignment") or 0) >= 4:
                all_strong.append(c)

    # Global highlights: best refusal + best answer wins across models
    refusals = sorted(
        [c for c in all_strong if c["pattern"] == "refusal"],
        key=_score_case,
        reverse=True,
    )
    gaps = sorted(
        [c for c in all_strong if c["pattern"] == "judge_gap"],
        key=_score_case,
        reverse=True,
    )
    highlights: list[dict[str, Any]] = []
    seen_hl: set[str] = set()
    for pool, limit in ((refusals, 8), (gaps, 8)):
        for c in pool:
            if c["case_id"] in seen_hl:
                continue
            # diversify models in highlights
            if sum(1 for h in highlights if h["model_label"] == c["model_label"]) >= 3:
                continue
            seen_hl.add(c["case_id"])
            highlights.append(_format_example(c))
            if len([h for h in highlights if h["pattern"] == c["pattern"]]) >= limit:
                break

    return {
        "schema": SCHEMA,
        "created_at": utc_iso(),
        "examples_per_model": examples_per_model,
        "statistics": stats,
        "triple_summary": catalog.get("summary", {}),
        "triple_by_pattern": triple_by_pattern,
        "highlights": highlights,
        "models": showcase_models,
    }


def _pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.1%}"


def _write_example_block(lines: list[str], ex: dict[str, Any], i: int) -> None:
    pat = "Refusal win" if ex["pattern"] == "refusal" else "Answer win"
    lines.extend(
        [
            f"### {i}. `{ex['eval_id']}` — {pat} ({ex['dataset']}) · {ex['model_label']}",
            "",
            f"**Q:** {ex['question']}",
            "",
            f"**Gold:** {ex['gold']}",
            "",
            f"**B3** (GA={ex['B3']['gold_alignment']}, G={ex['B3']['grounding']}):",
            "",
            str(ex["B3"]["pred"]),
            "",
            f"**B5** (GA={ex['B5']['gold_alignment']}, G={ex['B5']['grounding']}):",
            "",
            str(ex["B5"]["pred"]),
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


def _write_markdown(showcase: dict[str, Any], path: Path) -> None:
    stats = showcase["statistics"]
    model_order = [m["model_label"] for m in showcase["models"]]
    ms_by_label = {m["model_label"]: m for m in stats["per_model"]}

    lines = [
        "# Triple Hallucination Showcase — Ours vs B3 vs B5",
        "",
        "Curated examples where **Ours is correct** (GA≥4 or proper refusal) but "
        "**both B3 (uniform LoRA) and B5 (AdaLoRA) fail** (GA≤2 or invent on unanswerable gold).",
        "",
        f"**Generated:** {showcase['created_at']}  ",
        f"**Examples per model:** {showcase['examples_per_model']}  ",
        f"**Full catalog:** `triple_hallucination_catalog.json` ({showcase['triple_summary'].get('n_cases_total', '?')} cases)",
        "",
        "---",
        "",
        "## How to read this document",
        "",
        "| Pattern | Gold | B3 / B5 | Ours |",
        "|---------|------|---------|------|",
        "| **Refusal win** | Unanswerable (“not in document”) | Invents a plausible answer | Hedges / refuses |",
        "| **Answer win** | Answerable fact in context | Wrong / hallucinated (GA≤2) | Correct (GA≥4) |",
        "",
        "GA = gold alignment (Haiku judge, 1–5). G = grounding.",
        "",
        "**Use for thesis:** pick 2–3 refusal wins + 2–3 answer wins from the reference "
        "Llama-3.2-3B section, plus one scale contrast (e.g. 70B has almost no triple proofs).",
        "",
        "---",
        "",
        "## Executive summary",
        "",
        "1. **Triple proofs exist at every scale**, but are **most common on small/mid models** "
        "(Llama-3.2-3B: 589; Gemma-3-1B: 542) and **rare at 70B** (13).",
        "2. **Refusal discipline is the clearest QS win** on RepLiQA: B3/B5 invent on nearly all "
        "unanswerable questions; Ours correctly refuses on a non-trivial share (reference 3B).",
        "3. **Answer wins** show B3 and B5 often invent the *same wrong fact* (e.g. Apollo 13 vs 14) "
        "while Ours matches gold.",
        "4. **At 14B/70B**, B5 can have *lower* overall hallucination rate than Ours — consistent "
        "with B5 overtaking on mean GA at large scale — but **triple proofs still exist** "
        "(Ours uniquely correct on some questions).",
        "",
        "---",
        "",
        "## Hallucination statistics",
        "",
        "### Overall (all datasets pooled)",
        "",
        "Hallucination = invent on unanswerable gold, **or** GA≤2 on answerable gold.",
        "",
        "| Model | Eval rows | Halluc. B3 | B5 | Ours | Δ Ours−B3 | Δ Ours−B5 | Triple proofs |",
        "|-------|-----------|------------|-----|------|-----------|-----------|---------------|",
    ]

    for label in model_order:
        ms = ms_by_label.get(label)
        if not ms:
            continue
        hr = ms["hallucination_rate"]
        lines.append(
            f"| {ms['model_label']} | {ms['n_eval_rows']} | "
            f"{_pct(hr['B3'])} | {_pct(hr['B5'])} | {_pct(hr['Ours'])} | "
            f"{ms['hallucination_rate_delta_ours_vs_b3']:+.1%} | "
            f"{ms['hallucination_rate_delta_ours_vs_b5']:+.1%} | "
            f"{ms.get('triple_proof_n', 0)} |"
        )

    lines.extend(
        [
            "",
            "Positive Δ = Ours has **lower** hallucination rate than the baseline.",
            "",
            "### Per-dataset hallucination rate (Ours vs B3 / B5)",
            "",
        ]
    )

    for ds, ds_title in (
        ("repliqa", "RepLiQA (n=2,000)"),
        ("quoref", "Quoref (n=2,418)"),
        ("squad", "SQuAD v2 (n=11,873)"),
    ):
        lines.extend(
            [
                f"#### {ds_title}",
                "",
                "| Model | Halluc. B3 | B5 | Ours | Δ Ours−B3 | Δ Ours−B5 |",
                "|-------|------------|-----|------|-----------|-----------|",
            ]
        )
        for label in model_order:
            ms = ms_by_label.get(label)
            if not ms or ds not in ms.get("by_dataset", {}):
                continue
            d = ms["by_dataset"][ds]
            hr = d["hallucination_rate"]
            lines.append(
                f"| {label} | {_pct(hr['B3'])} | {_pct(hr['B5'])} | {_pct(hr['Ours'])} | "
                f"{d['hallucination_rate_delta_ours_vs_b3']:+.1%} | "
                f"{d['hallucination_rate_delta_ours_vs_b5']:+.1%} |"
            )
        lines.append("")

    lines.extend(
        [
            "### Refusal discipline (unanswerable gold only)",
            "",
            "Invent rate = fraction that invents instead of hedging/refusing. "
            "Quoref has no unanswerable gold (all answerable).",
            "",
            "| Model | Unans. n | Invent B3 | Invent B5 | Invent Ours | Correct refusal Ours |",
            "|-------|----------|-----------|-----------|-------------|----------------------|",
        ]
    )
    for label in model_order:
        ms = ms_by_label.get(label)
        if not ms:
            continue
        inv = ms["unanswerable_invent_rate"]
        ref = ms["refusal_correct_rate"]
        if inv.get("B3") is None:
            continue
        lines.append(
            f"| {label} | {ms.get('unanswerable_n', 0)} | "
            f"{_pct(inv['B3'])} | {_pct(inv['B5'])} | {_pct(inv['Ours'])} | "
            f"{_pct(ref['Ours'])} |"
        )

    # RepLiQA-only refusal (strongest story)
    lines.extend(
        [
            "",
            "#### RepLiQA only (unanswerable subset)",
            "",
            "| Model | Unans. n | Invent B3 | Invent B5 | Invent Ours | Correct refusal Ours |",
            "|-------|----------|-----------|-----------|-------------|----------------------|",
        ]
    )
    for label in model_order:
        ms = ms_by_label.get(label)
        if not ms or "repliqa" not in ms.get("by_dataset", {}):
            continue
        d = ms["by_dataset"]["repliqa"]
        inv = d["unanswerable_invent_rate"]
        ref = d["refusal_correct_rate"]
        if inv.get("B3") is None:
            continue
        lines.append(
            f"| {label} | {d.get('unanswerable_n', 0)} | "
            f"{_pct(inv['B3'])} | {_pct(inv['B5'])} | {_pct(inv['Ours'])} | "
            f"{_pct(ref['Ours'])} |"
        )

    lines.extend(
        [
            "",
            "### Triple-proof inventory (Ours GA≥4, B3+B5 fail)",
            "",
            "| Model | Refusal wins | Answer wins | Strong total | All catalog |",
            "|-------|--------------|-------------|--------------|-------------|",
        ]
    )
    tbp = showcase.get("triple_by_pattern", {})
    for label in model_order:
        p = tbp.get(label, {})
        lines.append(
            f"| {label} | {p.get('refusal', 0)} | {p.get('judge_gap', 0)} | "
            f"{p.get('strong_total', 0)} | {p.get('all_total', 0)} |"
        )

    # Highlights
    lines.extend(
        [
            "",
            "---",
            "",
            "## Highlights (best examples across models)",
            "",
            "Highest-scoring cases for slides / thesis figures. "
            "Full per-model lists follow.",
            "",
        ]
    )
    for i, ex in enumerate(showcase.get("highlights", []), start=1):
        _write_example_block(lines, ex, i)

    # Per-model sections
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
            _write_example_block(lines, ex, i)

    path.write_text("\n".join(lines), encoding="utf-8")


def run_export_triple_hallucination_showcase(ns: argparse.Namespace) -> int:
    catalog_path = Path(ns.catalog).expanduser().resolve()
    if catalog_path.is_file():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    else:
        catalog = build_catalog(cross_root=Path(ns.cross_root).expanduser().resolve())

    out_dir = Path(ns.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    showcase = build_showcase(
        catalog,
        examples_per_model=int(ns.examples_per_model),
        seed=int(ns.seed),
        cross_root=Path(ns.cross_root).expanduser().resolve(),
    )

    json_path = out_dir / "triple_hallucination_showcase.json"
    md_path = out_dir / "triple_hallucination_showcase.md"
    stats_path = out_dir / "triple_hallucination_stats.json"

    json_path.write_text(json.dumps(showcase, indent=2, ensure_ascii=False), encoding="utf-8")
    stats_path.write_text(
        json.dumps(
            {
                "schema": "triple_hallucination_stats/v1",
                "created_at": showcase["created_at"],
                **showcase["statistics"],
                "triple_summary": showcase["triple_summary"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_markdown(showcase, md_path)

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {stats_path}")
    return 0


def add_cli(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "export-triple-hallucination-showcase",
        help="Curated Ours-wins examples per model + hallucination stats (default 20)",
    )
    p.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=THESIS_ROOT / "experiments/analysis",
    )
    p.add_argument("--cross-root", type=Path, default=DEFAULT_CROSS_ROOT)
    p.add_argument("--examples-per-model", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=run_export_triple_hallucination_showcase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_cli(sub)
    raise SystemExit(parser.parse_args().fn(parser.parse_args()))
