"""
Compare fine-tuned conditions to a closed-model ceiling reference on judged GA.

Reads eval/judged/*/bedrock_judge_summary.json and reports:
  - mean gold_alignment per condition
  - gap_to_ceiling (ceiling GA - condition GA)
  - recovery_vs_b3: share of (B3 -> ceiling) gap closed by the condition

Usage:
  python -m thesis.cli eval-ceiling-gap --run-root /path/to/run
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.eval_repliqa_bedrock_judge import collect_existing_summaries

DEFAULT_CEILING = "REF_claude_opus"
DEFAULT_CEILINGS = ("REF_claude_opus", "REF_nova_2_lite")


def _ga(summary: dict[str, Any]) -> float | None:
    stats = summary.get("stats") or {}
    v = stats.get("mean_gold_alignment")
    return float(v) if v is not None else None


def compute_ceiling_gap(
    summaries: list[dict[str, Any]],
    *,
    ceiling: str,
    baseline: str | None = None,
) -> dict[str, Any]:
    by_cond: dict[str, dict[str, Any]] = {}
    for s in summaries:
        cond = str(s.get("condition") or "").strip()
        if not cond:
            continue
        by_cond[cond] = s

    if ceiling not in by_cond:
        raise ValueError(f"Ceiling condition {ceiling!r} not in judged summaries: {sorted(by_cond)}")

    ceiling_ga = _ga(by_cond[ceiling])
    b3_ga = _ga(by_cond[baseline]) if baseline and baseline in by_cond else None

    comparisons: list[dict[str, Any]] = []
    for cond, s in sorted(by_cond.items()):
        if cond == ceiling or cond.startswith("REF_"):
            continue
        ga = _ga(s)
        row: dict[str, Any] = {
            "condition": cond,
            "mean_gold_alignment": ga,
            "gap_to_ceiling": round(ceiling_ga - ga, 4) if ceiling_ga is not None and ga is not None else None,
        }
        if b3_ga is not None and ga is not None and ceiling_ga is not None:
            denom = ceiling_ga - b3_ga
            if abs(denom) > 1e-9:
                row["recovery_vs_b3"] = round((ga - b3_ga) / denom, 4)
                row["pct_of_ceiling_gain_vs_b3"] = round(100.0 * (ga - b3_ga) / denom, 1)
            else:
                row["recovery_vs_b3"] = None
                row["pct_of_ceiling_gain_vs_b3"] = None
        comparisons.append(row)

    comparisons.sort(
        key=lambda r: r.get("mean_gold_alignment") if r.get("mean_gold_alignment") is not None else -1.0,
        reverse=True,
    )
    return {
        "schema": "eval_ceiling_gap/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ceiling_condition": ceiling,
        "ceiling_mean_gold_alignment": ceiling_ga,
        "baseline_condition": baseline,
        "baseline_mean_gold_alignment": b3_ga,
        "comparisons": comparisons,
        "notes": [
            "gap_to_ceiling = ceiling GA - condition GA (lower is closer to best API answer).",
            "recovery_vs_b3 = (cond - B3) / (ceiling - B3); 1.0 means matches ceiling gain over B3.",
            "Judge is Bedrock Haiku (Anthropic). Ceilings: Opus (Anthropic) + Nova 2 Lite (Amazon).",
        ],
    }


def _default_output_name(ceiling: str) -> str:
    slug = ceiling.replace("REF_", "").replace("_", "-")
    return f"ceiling_gap_vs_{slug}.json"


def _auto_baseline(summaries: list[dict[str, Any]]) -> str | None:
    for cand in ("B3_lora_ctx", "B3_lora_all", "B3_lora_no_ctx"):
        if any(str(s.get("condition")) == cand for s in summaries):
            return cand
    return None


def _print_gap_table(doc: dict[str, Any]) -> None:
    ceiling = doc["ceiling_condition"]
    print(f"\n=== Ceiling gap (ref={ceiling}, GA={doc['ceiling_mean_gold_alignment']}) ===", flush=True)
    hdr = f"{'condition':<28} {'gold_al':>8} {'gap':>8} {'%ceil':>8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for row in doc["comparisons"]:
        if row["condition"] == ceiling or row["condition"].startswith("REF_"):
            continue
        pct = row.get("pct_of_ceiling_gain_vs_b3")
        print(
            f"{row['condition']:<28} "
            f"{row.get('mean_gold_alignment') if row.get('mean_gold_alignment') is not None else 'n/a':>8} "
            f"{row.get('gap_to_ceiling') if row.get('gap_to_ceiling') is not None else 'n/a':>8} "
            f"{pct if pct is not None else 'n/a':>8}",
            flush=True,
        )


def run_eval_ceiling_gap(ns: argparse.Namespace) -> int:
    run_root = Path(ns.run_root).expanduser().resolve()
    judged_dir = Path(ns.judged_dir).expanduser().resolve() if ns.judged_dir else run_root / "eval" / "judged"
    if not judged_dir.is_dir():
        print(f"Missing judged dir: {judged_dir}", file=sys.stderr)
        return 1

    summaries = collect_existing_summaries(judged_dir)
    if not summaries:
        print(f"No judge summaries under {judged_dir}", file=sys.stderr)
        return 1

    baseline = str(ns.baseline_condition) if ns.baseline_condition else _auto_baseline(summaries)

    if bool(getattr(ns, "all_ceilings", False)):
        ceilings = [c for c in DEFAULT_CEILINGS if any(str(s.get("condition")) == c for s in summaries)]
        if not ceilings:
            print(f"No REF ceilings judged yet (want {DEFAULT_CEILINGS})", file=sys.stderr)
            return 1
    else:
        ceilings = [str(ns.ceiling_condition or DEFAULT_CEILING)]

    reports: list[dict[str, Any]] = []
    for ceiling in ceilings:
        try:
            doc = compute_ceiling_gap(summaries, ceiling=ceiling, baseline=baseline)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
        reports.append(doc)
        out = (
            Path(ns.output_json).expanduser().resolve()
            if ns.output_json and len(ceilings) == 1
            else judged_dir / _default_output_name(ceiling)
        )
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _print_gap_table(doc)
        print(f"Wrote {out}", flush=True)

    if len(reports) > 1:
        summary = {
            "schema": "eval_ceiling_gap_summary/v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "baseline_condition": baseline,
            "ceilings": reports,
            "notes": [
                "Compare Ours/B3/B5 gap to Anthropic Opus and Amazon Nova 2 Lite ceilings.",
                "Haiku judge is Anthropic; Nova ceiling is cross-vendor on same Bedrock account.",
            ],
        }
        sum_path = judged_dir / "ceiling_gap_summary.json"
        sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {sum_path}", flush=True)

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Report GA gap vs closed-model ceiling reference.")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--judged-dir", type=Path, default=None)
    p.add_argument("--ceiling-condition", type=str, default=DEFAULT_CEILING)
    p.add_argument(
        "--all-ceilings",
        action="store_true",
        help="Report gap vs both REF_claude_sonnet and REF_gpt4o when judged.",
    )
    p.add_argument("--baseline-condition", type=str, default=None, help="Default: auto B3 if present.")
    p.add_argument("--output-json", type=Path, default=None)
    return p


if __name__ == "__main__":
    raise SystemExit(run_eval_ceiling_gap(build_arg_parser().parse_args()))
