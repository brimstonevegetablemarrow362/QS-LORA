"""
Curated examples where Ours avoids hallucination but a baseline does not.

Two patterns:
  1. answer_win   — answerable gold; Ours grounded (G≥4), baseline ungrounded (G≤2)
  2. refusal_win  — unanswerable gold; Ours correctly refuses, baseline invents

Comparisons: Ours vs B3 and Ours vs B5.

  python -m thesis.export_grounded_hallucination_examples
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
SCHEMA = "grounded_hallucination_examples/v1"
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


def _truncate(text: str | None, limit: int) -> str:
    t = str(text or "").strip()
    return t if len(t) <= limit else t[: limit - 3] + "..."


def _g(row: dict[str, Any]) -> float:
    try:
        return float(_judge_block(row).get("grounding") or 0)
    except (TypeError, ValueError):
        return 0.0


def _ga(row: dict[str, Any]) -> float:
    try:
        return float(_judge_block(row).get("gold_alignment") or 0)
    except (TypeError, ValueError):
        return 0.0


def _arm_block(row: dict[str, Any]) -> dict[str, Any]:
    j = _judge_block(row)
    pred = str(row.get("pred") or "")
    return {
        "pred": pred,
        "grounding": j.get("grounding"),
        "gold_alignment": j.get("gold_alignment"),
        "overall": j.get("overall"),
        "refusal_class": _classify_refusal(pred),
        "judge_reason": j.get("brief_reason"),
    }


def find_answer_wins(
    *,
    baseline_rows: dict[str, dict[str, Any]],
    ours_rows: dict[str, dict[str, Any]],
    baseline: Baseline,
    ours_min_g: float = 4.0,
    baseline_max_g: float = 2.0,
) -> list[dict[str, Any]]:
    """Answerable: Ours grounded, baseline ungrounded (fabricated / unsupported)."""
    hits: list[dict[str, Any]] = []
    for eid, bl in baseline_rows.items():
        if eid not in ours_rows:
            continue
        ours = ours_rows[eid]
        gold = str(bl.get("gold") or ours.get("gold") or "")
        if is_refusal_gold(gold):
            continue
        bp = str(bl.get("pred") or "")
        op = str(ours.get("pred") or "")
        if _g(bl) > baseline_max_g:
            continue
        if _g(ours) < ours_min_g:
            continue
        # Prefer Ours committed to a grounded answer (not a soft refuse on answerable).
        if _is_refusal_like(op):
            continue
        # Prefer clear grounded-and-correct Ours answers for showcase quality.
        if _ga(ours) < 4.0:
            continue
        # Prefer baseline that clearly invented / dumped unsupported content.
        if not _is_invented(bp) and _ga(bl) >= 3:
            continue
        # Prefer clear unsupported baseline (low gold alignment too).
        if _ga(bl) > 2.0:
            continue
        bl_b, ou_b = _arm_block(bl), _arm_block(ours)
        hits.append(
            {
                "eval_id": eid,
                "pattern": "answer_win",
                "question": bl.get("question") or ours.get("question"),
                "gold": gold,
                "context": bl.get("context") or ours.get("context"),
                baseline: bl_b,
                "Ours": ou_b,
                "_score": _g(ours) - _g(bl) + 0.25 * (_ga(ours) - _ga(bl)),
            }
        )
    hits.sort(key=lambda r: (-float(r["_score"]), str(r["eval_id"])))
    for r in hits:
        r.pop("_score", None)
    return hits


def find_refusal_wins(
    *,
    baseline_rows: dict[str, dict[str, Any]],
    ours_rows: dict[str, dict[str, Any]],
    baseline: Baseline,
    ours_min_ga: float = 4.0,
) -> list[dict[str, Any]]:
    """Unanswerable: Ours correctly refuses (GA≥4), baseline invents."""
    hits: list[dict[str, Any]] = []
    for eid, bl in baseline_rows.items():
        if eid not in ours_rows:
            continue
        ours = ours_rows[eid]
        gold = str(bl.get("gold") or ours.get("gold") or "")
        if not is_refusal_gold(gold):
            continue
        bp = str(bl.get("pred") or "")
        op = str(ours.get("pred") or "")
        if not (_is_refusal_like(op) and _is_invented(bp)):
            continue
        # Require a *correct* refusal from Ours (aligned with gold "not found").
        if _ga(ours) < ours_min_ga:
            continue
        bl_b, ou_b = _arm_block(bl), _arm_block(ours)
        hits.append(
            {
                "eval_id": eid,
                "pattern": "refusal_win",
                "question": bl.get("question") or ours.get("question"),
                "gold": gold,
                "context": bl.get("context") or ours.get("context"),
                baseline: bl_b,
                "Ours": ou_b,
                "_score": _ga(ours) - _ga(bl),
            }
        )
    hits.sort(key=lambda r: (-float(r["_score"]), str(r["eval_id"])))
    for r in hits:
        r.pop("_score", None)
    return hits


def _collect(
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
    # Paired on shared eval_ids.
    common = set(bl_rows) & set(ours_rows)
    bl_rows = {k: bl_rows[k] for k in common}
    ours_rows = {k: ours_rows[k] for k in common}
    meta = MODEL_META[model_slug]
    cases: list[dict[str, Any]] = []
    for row in find_refusal_wins(
        baseline_rows=bl_rows, ours_rows=ours_rows, baseline=baseline
    ) + find_answer_wins(
        baseline_rows=bl_rows, ours_rows=ours_rows, baseline=baseline
    ):
        cases.append(
            {
                "case_id": f"{model_slug}/{dataset}/{baseline}/{row['pattern']}/{row['eval_id']}",
                "eval_id": row["eval_id"],
                "dataset": dataset,
                "pattern": row["pattern"],
                "baseline": baseline,
                "model_slug": model_slug,
                "model_label": meta["label"],
                "source": source,
                "baseline_condition": bl_cond,
                "ours_condition": ours_cond,
                "question": row["question"],
                "gold": row["gold"],
                "context": row["context"],
                baseline: row[baseline],
                "Ours": row["Ours"],
            }
        )
    return cases


def build_catalog(baseline: Baseline, *, cross_root: Path) -> dict[str, Any]:
    all_cases: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []

    def add(model_slug: str, dataset: str, run_root: Path, source: str) -> None:
        cases = _collect(
            model_slug=model_slug,
            dataset=dataset,
            run_root=run_root,
            source=source,
            baseline=baseline,
        )
        if not cases:
            return
        per_run.append(
            {
                "model_slug": model_slug,
                "model_label": MODEL_META[model_slug]["label"],
                "dataset": dataset,
                "source": source,
                "n_refusal_win": sum(1 for c in cases if c["pattern"] == "refusal_win"),
                "n_answer_win": sum(1 for c in cases if c["pattern"] == "answer_win"),
                "n_total": len(cases),
            }
        )
        all_cases.extend(cases)

    for model_slug, dataset, run_root, source in REFERENCE_RUNS:
        add(model_slug, dataset, run_root, source)
    for model_slug in MODEL_META:
        if model_slug == "llama32_3b":
            continue
        for dataset, ds_dir in CROSS_DATASET_DIRS.items():
            run_root = cross_root / model_slug / ds_dir
            if run_root.is_dir():
                add(model_slug, dataset, run_root, "cross_model")

    by_pattern = defaultdict(int)
    by_dataset = defaultdict(int)
    by_model = defaultdict(int)
    for c in all_cases:
        by_pattern[c["pattern"]] += 1
        by_dataset[c["dataset"]] += 1
        by_model[c["model_label"]] += 1

    return {
        "schema": f"grounded_hallucination_catalog/v1",
        "baseline": baseline,
        "created_at": utc_iso(),
        "description": (
            f"Grounding-aligned Ours wins vs {baseline}: answer_win (Ours G≥4, "
            f"{baseline} G≤2 on answerable) and refusal_win (Ours refuses, "
            f"{baseline} invents on unanswerable)."
        ),
        "summary": {
            "n_cases_total": len(all_cases),
            "n_runs": len(per_run),
            "by_pattern": dict(sorted(by_pattern.items())),
            "by_dataset": dict(sorted(by_dataset.items())),
            "by_model_label": dict(sorted(by_model.items())),
            "per_run": per_run,
        },
        "cases": all_cases,
    }


def _score(case: dict[str, Any], baseline: Baseline) -> float:
    bl = case[baseline]
    ours = case["Ours"]
    if case["pattern"] == "refusal_win":
        return float(ours.get("gold_alignment") or 0) - float(bl.get("gold_alignment") or 0)
    return float(ours.get("grounding") or 0) - float(bl.get("grounding") or 0)


def _pick(
    cases: list[dict[str, Any]],
    baseline: Baseline,
    *,
    n_refusal: int = 10,
    n_answer: int = 12,
) -> list[dict[str, Any]]:
    refusals = [c for c in cases if c["pattern"] == "refusal_win"]
    answers = [c for c in cases if c["pattern"] == "answer_win"]
    refusals.sort(key=lambda c: _score(c, baseline), reverse=True)
    answers.sort(key=lambda c: _score(c, baseline), reverse=True)

    picked: list[dict[str, Any]] = []
    seen: set[str] = set()

    def take(pool: list[dict[str, Any]], limit: int) -> None:
        # Round-robin across models/datasets for diversity.
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for c in pool:
            buckets[(c["model_label"], c["dataset"])].append(c)
        for lst in buckets.values():
            lst.sort(key=lambda c: _score(c, baseline), reverse=True)
        keys = sorted(buckets.keys(), key=lambda k: (MODEL_ORDER.index(k[0]) if k[0] in MODEL_ORDER else 99, k[1]))
        while sum(1 for c in picked if c["pattern"] == pool[0]["pattern"]) < limit and any(buckets.values()):
            progressed = False
            for key in keys:
                if sum(1 for c in picked if c["pattern"] == pool[0]["pattern"]) >= limit:
                    break
                while buckets[key] and buckets[key][0]["case_id"] in seen:
                    buckets[key].pop(0)
                if not buckets[key]:
                    continue
                # Cap 2 per (model, dataset) so one model can't dominate.
                if sum(1 for c in picked if (c["model_label"], c["dataset"]) == key and c["pattern"] == pool[0]["pattern"]) >= 2:
                    buckets[key].pop(0)
                    continue
                c = buckets[key].pop(0)
                seen.add(c["case_id"])
                picked.append(c)
                progressed = True
            if not progressed:
                break

    if refusals:
        take(refusals, n_refusal)
    if answers:
        take(answers, n_answer)
    # Prefer refusal examples first in the display order.
    picked.sort(
        key=lambda c: (
            0 if c["pattern"] == "refusal_win" else 1,
            MODEL_ORDER.index(c["model_label"]) if c["model_label"] in MODEL_ORDER else 99,
            c["dataset"],
            -_score(c, baseline),
        )
    )
    return picked


def _fmt_example(case: dict[str, Any], baseline: Baseline, idx: int) -> list[str]:
    bl = case[baseline]
    ours = case["Ours"]
    pat = "Correct refusal (Ours abstains; baseline invents)" if case["pattern"] == "refusal_win" else "Grounded answer (Ours grounded; baseline unsupported)"
    lines = [
        f"### Example {idx} — {case['model_label']} / {case['dataset']} — {pat}",
        "",
        f"**eval_id:** `{case['eval_id']}`  ",
        f"**Question:** {case.get('question')}",
        "",
        f"**Gold:** {_truncate(case.get('gold'), 280)}",
        "",
        f"**Context (preview):** {_truncate(case.get('context'), 450)}",
        "",
        f"**{baseline}** (G={bl.get('grounding')}, GA={bl.get('gold_alignment')}, class=`{bl.get('refusal_class')}`):",
        "",
        f"> {_truncate(bl.get('pred'), 500)}",
        "",
    ]
    if bl.get("judge_reason"):
        lines.extend([f"*Judge:* {_truncate(bl.get('judge_reason'), 220)}", ""])
    lines.extend(
        [
            f"**Ours** (G={ours.get('grounding')}, GA={ours.get('gold_alignment')}, class=`{ours.get('refusal_class')}`):",
            "",
            f"> {_truncate(ours.get('pred'), 500)}",
            "",
        ]
    )
    if ours.get("judge_reason"):
        lines.extend([f"*Judge:* {_truncate(ours.get('judge_reason'), 220)}", ""])
    lines.append("---")
    lines.append("")
    return lines


def write_markdown(
    *,
    catalogs: dict[Baseline, dict[str, Any]],
    showcases: dict[Baseline, list[dict[str, Any]]],
    out_path: Path,
) -> None:
    lines = [
        "# Hallucination examples — Ours does not, baseline does",
        "",
        f"**Generated:** {utc_iso()}  ",
        "**Criterion:** grounding-aligned (matches §8 of `HALLUCINATION_REPORT.md`).",
        "",
        "Two patterns:",
        "",
        "1. **Refusal win** — gold is unanswerable; **Ours correctly refuses**, baseline **invents** an answer.",
        "2. **Answer win** — gold is answerable; **Ours is grounded** (judge G≥4), baseline is **unsupported** (G≤2).",
        "",
        "Comparisons are pairwise: **Ours vs B3** and **Ours vs B5** (same question).",
        "",
        "---",
        "",
        "## Catalog counts",
        "",
        "| Comparison | Total cases | Refusal wins | Answer wins |",
        "|------------|-------------|--------------|-------------|",
    ]
    for bl in ("B3", "B5"):
        s = catalogs[bl]["summary"]
        bp = s["by_pattern"]
        lines.append(
            f"| Ours vs {bl} | {s['n_cases_total']} | {bp.get('refusal_win', 0)} | {bp.get('answer_win', 0)} |"
        )

    lines.extend(["", "### Refusal-win counts by model (Ours abstains; baseline invents)", ""])
    lines.append("| Model | vs B3 | vs B5 |")
    lines.append("|-------|-------|-------|")
    for label in MODEL_ORDER:
        n3 = sum(
            1
            for c in catalogs["B3"]["cases"]
            if c["model_label"] == label and c["pattern"] == "refusal_win"
        )
        n5 = sum(
            1
            for c in catalogs["B5"]["cases"]
            if c["model_label"] == label and c["pattern"] == "refusal_win"
        )
        if n3 or n5:
            lines.append(f"| {label} | {n3} | {n5} |")

    for bl in ("B3", "B5"):
        examples = showcases[bl]
        n_ref = sum(1 for e in examples if e["pattern"] == "refusal_win")
        n_ans = sum(1 for e in examples if e["pattern"] == "answer_win")
        lines.extend(
            [
                "",
                "---",
                "",
                f"## Ours vs {bl} — curated examples ({n_ref} refusal + {n_ans} answer wins)",
                "",
            ]
        )
        # Refusal section
        lines.append(f"### Refusal wins — Ours refuses, {bl} invents")
        lines.append("")
        ref_ex = [e for e in examples if e["pattern"] == "refusal_win"]
        if not ref_ex:
            lines.append("*No refusal-win examples found for this baseline (models rarely abstain).*")
            lines.append("")
        else:
            for i, ex in enumerate(ref_ex, start=1):
                lines.extend(_fmt_example(ex, bl, i))
        lines.append(f"### Answer wins — Ours grounded, {bl} unsupported")
        lines.append("")
        ans_ex = [e for e in examples if e["pattern"] == "answer_win"]
        for i, ex in enumerate(ans_ex, start=1):
            lines.extend(_fmt_example(ex, bl, i))

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "```",
            "experiments/analysis/",
            "├── HALLUCINATION_EXAMPLES.md                  # this file",
            "├── grounded_hallucination_examples_vs_b3.json",
            "└── grounded_hallucination_examples_vs_b5.json",
            "```",
            "",
            "Generator: `python -m thesis.export_grounded_hallucination_examples`",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cross-root", type=Path, default=DEFAULT_CROSS_ROOT)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=THESIS_ROOT / "experiments" / "analysis",
    )
    ap.add_argument("--n-refusal", type=int, default=10)
    ap.add_argument("--n-answer", type=int, default=12)
    ns = ap.parse_args()
    out_dir = ns.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    catalogs: dict[Baseline, dict[str, Any]] = {}
    showcases: dict[Baseline, list[dict[str, Any]]] = {}
    for bl in ("B3", "B5"):
        print(f"Building catalog Ours vs {bl}...", flush=True)
        cat = build_catalog(bl, cross_root=ns.cross_root)
        catalogs[bl] = cat
        picks = _pick(cat["cases"], bl, n_refusal=ns.n_refusal, n_answer=ns.n_answer)
        showcases[bl] = picks
        # Compact JSON for the curated picks only (catalog can be huge).
        slim = {
            "schema": SCHEMA,
            "baseline": bl,
            "created_at": utc_iso(),
            "catalog_summary": cat["summary"],
            "examples": [
                {
                    **{k: v for k, v in e.items() if k != "context"},
                    "context": _truncate(e.get("context"), 800),
                    e["baseline"]: {
                        **e[e["baseline"]],
                        "pred": _truncate(e[e["baseline"]].get("pred"), 600),
                        "judge_reason": _truncate(e[e["baseline"]].get("judge_reason"), 300),
                    },
                    "Ours": {
                        **e["Ours"],
                        "pred": _truncate(e["Ours"].get("pred"), 600),
                        "judge_reason": _truncate(e["Ours"].get("judge_reason"), 300),
                    },
                }
                for e in picks
            ],
        }
        out_json = out_dir / f"grounded_hallucination_examples_vs_{bl.lower()}.json"
        out_json.write_text(json.dumps(slim, indent=2), encoding="utf-8")
        print(
            f"  {bl}: {cat['summary']['n_cases_total']} cases "
            f"(refusal={cat['summary']['by_pattern'].get('refusal_win', 0)}, "
            f"answer={cat['summary']['by_pattern'].get('answer_win', 0)}); "
            f"showcased {len(picks)} → {out_json}",
            flush=True,
        )

    md_path = out_dir / "HALLUCINATION_EXAMPLES.md"
    # Also mirror next to the main report for easy discovery.
    mirror = THESIS_ROOT / "HALLUCINATION_EXAMPLES.md"
    write_markdown(catalogs=catalogs, showcases=showcases, out_path=md_path)
    mirror.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {mirror}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
