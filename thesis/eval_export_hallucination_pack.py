"""
Export qualitative examples: unanswerable gold, Ours hedges/refuses, B3 invents an answer.

Outputs JSONL + Markdown under eval/study_samples/.

  python -m thesis.cli eval-export-hallucination-pack --run-root .../repliqa_train_0-3
  python -m thesis.cli eval-export-hallucination-pack --run-root .../squad_qa_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.qa_answer_metrics import is_invalid_answer, is_refusal_gold


def _load_jsonl_map(path: Path, *, pred_key: str = "pred") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        eid = str(row.get("eval_id") or row.get("chunk_id") or "").strip()
        if eid:
            out[eid] = row
    return out


def _classify_refusal(pred: str) -> str:
    p = (pred or "").strip()
    pl = p.lower()
    if is_invalid_answer(p):
        return "empty"
    if "not found in the document" in pl:
        return "canonical_refusal"
    if any(
        x in pl
        for x in (
            "cannot be answered",
            "unanswerable",
            "no answer in",
        )
    ):
        return "explicit_refusal"
    if any(
        x in pl[:160]
        for x in (
            "does not mention",
            "does not provide",
            "does not explicitly",
            "does not specifically",
            "does not contain",
            "not in the document",
            "no information",
            "not provide",
        )
    ):
        return "soft_refusal"
    if len(p) > 150:
        return "context_dump"
    return "invented_answer"


def _is_invented(pred: str) -> bool:
    return _classify_refusal(pred) in ("invented_answer", "context_dump")


def _is_refusal_like(pred: str) -> bool:
    return _classify_refusal(pred) in ("canonical_refusal", "explicit_refusal", "soft_refusal")


def _judge_block(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("llm_judge") or {}


def find_refusal_vs_invent_cases(
    *,
    b3_rows: dict[str, dict[str, Any]],
    ours_rows: dict[str, dict[str, Any]],
    require_judge: bool = False,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for eid, b3 in b3_rows.items():
        if eid not in ours_rows:
            continue
        ours = ours_rows[eid]
        gold = str(b3.get("gold") or ours.get("gold") or "")
        if not is_refusal_gold(gold):
            continue
        bp = str(b3.get("pred") or "")
        op = str(ours.get("pred") or "")
        if not (_is_refusal_like(op) and _is_invented(bp)):
            continue
        bj, oj = _judge_block(b3), _judge_block(ours)
        if require_judge and not oj.get("gold_alignment"):
            continue
        score = float(oj.get("gold_alignment") or 0) - float(bj.get("gold_alignment") or 0)
        hits.append(
            {
                "eval_id": eid,
                "question": b3.get("question") or ours.get("question"),
                "gold": gold,
                "b3_pred": bp,
                "ours_pred": op,
                "b3_refusal_class": _classify_refusal(bp),
                "ours_refusal_class": _classify_refusal(op),
                "b3_gold_alignment": bj.get("gold_alignment"),
                "ours_gold_alignment": oj.get("gold_alignment"),
                "b3_grounding": bj.get("grounding"),
                "ours_grounding": oj.get("grounding"),
                "b3_judge_reason": bj.get("brief_reason"),
                "ours_judge_reason": oj.get("brief_reason"),
                "context_preview": (str(b3.get("context") or ours.get("context") or ""))[:800],
                "_score": score,
            }
        )
    hits.sort(key=lambda r: (-float(r["_score"]), str(r["eval_id"])))
    for row in hits:
        row.pop("_score", None)
    return hits


def find_judge_gap_cases(
    *,
    b3_rows: dict[str, dict[str, Any]],
    ours_rows: dict[str, dict[str, Any]],
    require_judge: bool = False,
    baseline_max_ga: float = 2.0,
    ours_min_ga: float = 4.0,
    min_gap: float = 2.0,
) -> list[dict[str, Any]]:
    """Answerable QA: baseline wrong/hallucinated (low GA), Ours aligned with gold (high GA)."""
    hits: list[dict[str, Any]] = []
    for eid, b3 in b3_rows.items():
        if eid not in ours_rows:
            continue
        ours = ours_rows[eid]
        gold = str(b3.get("gold") or ours.get("gold") or "")
        if is_refusal_gold(gold):
            continue
        bj, oj = _judge_block(b3), _judge_block(ours)
        if require_judge and (bj.get("gold_alignment") is None or oj.get("gold_alignment") is None):
            continue
        bga = float(bj.get("gold_alignment") or 0)
        oga = float(oj.get("gold_alignment") or 0)
        if bga > baseline_max_ga or oga < ours_min_ga or (oga - bga) < min_gap:
            continue
        bp = str(b3.get("pred") or "")
        op = str(ours.get("pred") or "")
        hits.append(
            {
                "eval_id": eid,
                "question": b3.get("question") or ours.get("question"),
                "gold": gold,
                "b3_pred": bp,
                "ours_pred": op,
                "b3_refusal_class": _classify_refusal(bp),
                "ours_refusal_class": _classify_refusal(op),
                "b3_gold_alignment": bj.get("gold_alignment"),
                "ours_gold_alignment": oj.get("gold_alignment"),
                "b3_grounding": bj.get("grounding"),
                "ours_grounding": oj.get("grounding"),
                "b3_judge_reason": bj.get("brief_reason"),
                "ours_judge_reason": oj.get("brief_reason"),
                "context_preview": (str(b3.get("context") or ours.get("context") or ""))[:800],
                "_score": oga - bga,
            }
        )
    hits.sort(key=lambda r: (-float(r["_score"]), str(r["eval_id"])))
    for row in hits:
        row.pop("_score", None)
    return hits


def find_triple_judge_gap_cases(
    *,
    b3_rows: dict[str, dict[str, Any]],
    b5_rows: dict[str, dict[str, Any]],
    ours_rows: dict[str, dict[str, Any]],
    baseline_max_ga: float = 2.0,
    ours_min_ga: float = 4.0,
    min_gap: float = 2.0,
) -> list[dict[str, Any]]:
    """Answerable gold: both baselines wrong (GA≤2), Ours correct (GA≥4)."""
    hits: list[dict[str, Any]] = []
    for eid, ours in ours_rows.items():
        if eid not in b3_rows or eid not in b5_rows:
            continue
        b3, b5 = b3_rows[eid], b5_rows[eid]
        gold = str(ours.get("gold") or b3.get("gold") or b5.get("gold") or "")
        if is_refusal_gold(gold):
            continue
        oj = _judge_block(ours)
        b3j, b5j = _judge_block(b3), _judge_block(b5)
        oga = float(oj.get("gold_alignment") or 0)
        b3ga = float(b3j.get("gold_alignment") or 0)
        b5ga = float(b5j.get("gold_alignment") or 0)
        if oga < ours_min_ga or b3ga > baseline_max_ga or b5ga > baseline_max_ga:
            continue
        if (oga - b3ga) < min_gap or (oga - b5ga) < min_gap:
            continue
        hits.append(
            {
                "eval_id": eid,
                "question": ours.get("question") or b3.get("question"),
                "gold": gold,
                "b3_pred": str(b3.get("pred") or ""),
                "b5_pred": str(b5.get("pred") or ""),
                "ours_pred": str(ours.get("pred") or ""),
                "b3_gold_alignment": b3j.get("gold_alignment"),
                "b5_gold_alignment": b5j.get("gold_alignment"),
                "ours_gold_alignment": oj.get("gold_alignment"),
                "b3_grounding": b3j.get("grounding"),
                "b5_grounding": b5j.get("grounding"),
                "ours_grounding": oj.get("grounding"),
                "b3_refusal_class": _classify_refusal(str(b3.get("pred") or "")),
                "b5_refusal_class": _classify_refusal(str(b5.get("pred") or "")),
                "ours_refusal_class": _classify_refusal(str(ours.get("pred") or "")),
                "b3_judge_reason": b3j.get("brief_reason"),
                "b5_judge_reason": b5j.get("brief_reason"),
                "ours_judge_reason": oj.get("brief_reason"),
                "context_preview": (str(ours.get("context") or ""))[:800],
                "_score": min(oga - b3ga, oga - b5ga),
            }
        )
    hits.sort(key=lambda r: (-float(r["_score"]), str(r["eval_id"])))
    for row in hits:
        row.pop("_score", None)
    return hits


def find_triple_refusal_cases(
    *,
    b3_rows: dict[str, dict[str, Any]],
    b5_rows: dict[str, dict[str, Any]],
    ours_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Unanswerable gold: Ours hedges/refuses; B3 and B5 both invent."""
    hits: list[dict[str, Any]] = []
    for eid, ours in ours_rows.items():
        if eid not in b3_rows or eid not in b5_rows:
            continue
        b3, b5 = b3_rows[eid], b5_rows[eid]
        gold = str(ours.get("gold") or b3.get("gold") or b5.get("gold") or "")
        if not is_refusal_gold(gold):
            continue
        op = str(ours.get("pred") or "")
        bp = str(b3.get("pred") or "")
        b5p = str(b5.get("pred") or "")
        if not (_is_refusal_like(op) and _is_invented(bp) and _is_invented(b5p)):
            continue
        oj = _judge_block(ours)
        b3j, b5j = _judge_block(b3), _judge_block(b5)
        score = float(oj.get("gold_alignment") or 0) - max(
            float(b3j.get("gold_alignment") or 0),
            float(b5j.get("gold_alignment") or 0),
        )
        hits.append(
            {
                "eval_id": eid,
                "question": ours.get("question") or b3.get("question"),
                "gold": gold,
                "b3_pred": bp,
                "b5_pred": b5p,
                "ours_pred": op,
                "b3_gold_alignment": b3j.get("gold_alignment"),
                "b5_gold_alignment": b5j.get("gold_alignment"),
                "ours_gold_alignment": oj.get("gold_alignment"),
                "b3_grounding": b3j.get("grounding"),
                "b5_grounding": b5j.get("grounding"),
                "ours_grounding": oj.get("grounding"),
                "b3_refusal_class": _classify_refusal(bp),
                "b5_refusal_class": _classify_refusal(b5p),
                "ours_refusal_class": _classify_refusal(op),
                "b3_judge_reason": b3j.get("brief_reason"),
                "b5_judge_reason": b5j.get("brief_reason"),
                "ours_judge_reason": oj.get("brief_reason"),
                "context_preview": (str(ours.get("context") or ""))[:800],
                "_score": score,
            }
        )
    hits.sort(key=lambda r: (-float(r["_score"]), str(r["eval_id"])))
    for row in hits:
        row.pop("_score", None)
    return hits


def _write_triple_markdown(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    title: str,
    b3_condition: str,
    b5_condition: str,
    ours_condition: str,
    mode: str,
) -> None:
    if mode == "judge_gap":
        pattern = (
            f"**Pattern:** answerable gold; **{b3_condition}** and **{b5_condition}** "
            f"wrong/hallucinated (GA≤2); **{ours_condition}** matches gold (GA≥4)."
        )
    else:
        pattern = (
            f"**Pattern:** unanswerable gold; **{ours_condition}** hedges/refuses; "
            f"**{b3_condition}** and **{b5_condition}** both invent answers."
        )
    lines = [
        f"# {title}",
        "",
        pattern,
        f"**N:** {len(rows)}  ",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "---",
        "",
    ]
    for i, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {i}. `{row['eval_id']}`",
                "",
                f"**Q:** {row['question']}",
                "",
                f"**Gold:** {row['gold']}",
                "",
                f"### B3 `{b3_condition}` (GA={row.get('b3_gold_alignment')}, G={row.get('b3_grounding')}, class={row.get('b3_refusal_class')})",
                "",
                str(row["b3_pred"]),
                "",
                f"### B5 `{b5_condition}` (GA={row.get('b5_gold_alignment')}, G={row.get('b5_grounding')}, class={row.get('b5_refusal_class')})",
                "",
                str(row["b5_pred"]),
                "",
                f"### Ours `{ours_condition}` (GA={row.get('ours_gold_alignment')}, G={row.get('ours_grounding')}, class={row.get('ours_refusal_class')})",
                "",
                str(row["ours_pred"]),
                "",
            ]
        )
        if row.get("ours_judge_reason"):
            lines.extend([f"*Judge (Ours):* {row['ours_judge_reason']}", ""])
        lines.append("---")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_triple_hallucination_pack(
    *,
    run_root: Path,
    b3_condition: str,
    b5_condition: str,
    ours_condition: str,
    output_dir: Path | None = None,
    max_examples: int = 25,
    mode: str = "auto",
) -> tuple[Path, Path, int, str]:
    eval_dir = run_root / "eval"
    out_dir = output_dir or eval_dir / "study_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _judged(cond: str) -> dict[str, dict[str, Any]]:
        path = eval_dir / "judged" / cond / "bedrock_judge.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        return _load_jsonl_map(path)

    b3_rows = _judged(b3_condition)
    b5_rows = _judged(b5_condition)
    ours_rows = _judged(ours_condition)

    resolved_mode = str(mode).lower()
    if resolved_mode == "auto":
        rows = find_triple_refusal_cases(b3_rows=b3_rows, b5_rows=b5_rows, ours_rows=ours_rows)
        resolved_mode = "refusal" if rows else "judge_gap"
    if resolved_mode == "refusal":
        rows = find_triple_refusal_cases(b3_rows=b3_rows, b5_rows=b5_rows, ours_rows=ours_rows)
    elif resolved_mode == "judge_gap":
        rows = find_triple_judge_gap_cases(
            b3_rows=b3_rows, b5_rows=b5_rows, ours_rows=ours_rows
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    if max_examples > 0:
        rows = rows[:max_examples]

    prefix = "triple_refusal" if resolved_mode == "refusal" else "triple_hallucination_gap"
    stem = f"{prefix}_{b3_condition}_vs_{b5_condition}_vs_{ours_condition}_{len(rows)}"
    jsonl_path = out_dir / f"{stem}.jsonl"
    md_path = out_dir / f"{stem}.md"

    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    title = (
        "Triple refusal gap (B3+B5 invent, Ours hedges)"
        if resolved_mode == "refusal"
        else "Triple hallucination gap (B3+B5 wrong, Ours correct)"
    )
    _write_triple_markdown(
        rows,
        md_path,
        title=f"{title} — {run_root.name}",
        b3_condition=b3_condition,
        b5_condition=b5_condition,
        ours_condition=ours_condition,
        mode=resolved_mode,
    )
    return jsonl_path, md_path, len(rows), resolved_mode


def run_eval_export_triple_hallucination_pack(ns: argparse.Namespace) -> int:
    run_root = Path(ns.run_root).expanduser().resolve()
    jsonl_path, md_path, n, mode = export_triple_hallucination_pack(
        run_root=run_root,
        b3_condition=str(ns.b3_condition),
        b5_condition=str(ns.b5_condition),
        ours_condition=str(ns.ours_condition),
        output_dir=Path(ns.output_dir).expanduser().resolve() if ns.output_dir else None,
        max_examples=int(ns.max_examples),
        mode=str(ns.mode),
    )
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {md_path}")
    print(f"  n={n}  mode={mode}")
    return 0


def _pattern_description(mode: str, baseline: str, ours: str) -> str:
    if mode == "judge_gap":
        return (
            f"**Pattern:** answerable gold; **{baseline}** wrong or hallucinated (GA≤2); "
            f"**{ours}** matches gold (GA≥4)."
        )
    return (
        f"**Pattern:** gold = unanswerable; **{ours}** hedges/refuses; **{baseline}** invents an answer."
    )


def _write_markdown(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    title: str,
    baseline: str,
    ours: str,
    mode: str = "refusal",
) -> None:
    lines = [
        f"# {title}",
        "",
        _pattern_description(mode, baseline, ours),
        f"**N:** {len(rows)}  ",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "---",
        "",
    ]
    for i, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {i}. `{row['eval_id']}`",
                "",
                f"**Q:** {row['question']}",
                "",
                f"**Gold:** {row['gold']}",
                "",
                f"### Baseline `{baseline}` (GA={row.get('b3_gold_alignment')}, class={row.get('b3_refusal_class')})",
                "",
                str(row["b3_pred"]),
                "",
                f"### Ours `{ours}` (GA={row.get('ours_gold_alignment')}, class={row.get('ours_refusal_class')})",
                "",
                str(row["ours_pred"]),
                "",
            ]
        )
        if row.get("ours_judge_reason"):
            lines.extend([f"*Judge (Ours):* {row['ours_judge_reason']}", ""])
        lines.append("---")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_hallucination_pack(
    *,
    run_root: Path,
    b3_condition: str,
    ours_condition: str,
    output_dir: Path | None = None,
    max_examples: int = 0,
    use_judged: bool = True,
    mode: str = "auto",
) -> tuple[Path, Path, int, str]:
    eval_dir = run_root / "eval"
    out_dir = output_dir or eval_dir / "study_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_judged:
        b3_path = eval_dir / "judged" / b3_condition / "bedrock_judge.jsonl"
        ours_path = eval_dir / "judged" / ours_condition / "bedrock_judge.jsonl"
    else:
        b3_path = eval_dir / "predictions" / b3_condition / "predictions.jsonl"
        ours_path = eval_dir / "predictions" / ours_condition / "predictions.jsonl"

    for p in (b3_path, ours_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    b3_rows = _load_jsonl_map(b3_path)
    ours_rows = _load_jsonl_map(ours_path)
    resolved_mode = str(mode).lower()
    if resolved_mode == "auto":
        rows = find_refusal_vs_invent_cases(
            b3_rows=b3_rows,
            ours_rows=ours_rows,
            require_judge=use_judged,
        )
        resolved_mode = "refusal" if rows else "judge_gap"
    if resolved_mode == "refusal":
        rows = find_refusal_vs_invent_cases(
            b3_rows=b3_rows,
            ours_rows=ours_rows,
            require_judge=use_judged,
        )
    elif resolved_mode == "judge_gap":
        rows = find_judge_gap_cases(
            b3_rows=b3_rows,
            ours_rows=ours_rows,
            require_judge=use_judged,
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}; use refusal, judge_gap, or auto")

    if max_examples > 0:
        rows = rows[:max_examples]

    prefix = "refusal_vs_invent" if resolved_mode == "refusal" else "hallucination_gap"
    stem = f"{prefix}_{b3_condition}_vs_{ours_condition}_{len(rows)}"
    jsonl_path = out_dir / f"{stem}.jsonl"
    md_path = out_dir / f"{stem}.md"

    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    title = (
        "Refusal vs invented answer"
        if resolved_mode == "refusal"
        else "Hallucination gap (baseline wrong, Ours correct)"
    )
    _write_markdown(
        rows,
        md_path,
        title=f"{title} — {run_root.name}",
        baseline=b3_condition,
        ours=ours_condition,
        mode=resolved_mode,
    )
    return jsonl_path, md_path, len(rows), resolved_mode


def run_eval_export_hallucination_pack(ns: argparse.Namespace) -> int:
    run_root = Path(ns.run_root).expanduser().resolve()
    jsonl_path, md_path, n, mode = export_hallucination_pack(
        run_root=run_root,
        b3_condition=str(ns.b3_condition),
        ours_condition=str(ns.ours_condition),
        output_dir=Path(ns.output_dir).expanduser().resolve() if ns.output_dir else None,
        max_examples=int(ns.max_examples),
        use_judged=not bool(ns.from_predictions),
        mode=str(getattr(ns, "mode", "auto")),
    )
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {md_path}")
    print(f"  n={n}  mode={mode}  baseline={ns.b3_condition}  ours={ns.ours_condition}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export Ours-refusal vs B3-invent examples.")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--b3-condition", type=str, default="B3_lora_ctx")
    p.add_argument("--ours-condition", type=str, default="Ours_tier_ctx")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-examples", type=int, default=0, help="0 = all")
    p.add_argument(
        "--from-predictions",
        action="store_true",
        help="Use eval/predictions instead of eval/judged",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=("auto", "refusal", "judge_gap"),
        default="auto",
        help="auto: refusal if unanswerable gold exists, else judge_gap (Quoref)",
    )
    return p


if __name__ == "__main__":
    raise SystemExit(run_eval_export_hallucination_pack(build_arg_parser().parse_args()))
