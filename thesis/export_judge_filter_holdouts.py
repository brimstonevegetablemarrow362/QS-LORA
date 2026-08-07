"""
Export per-domain judge-filter eval holdouts for multidomain distillation eval.

  python -m thesis.cli export-judge-filter-holdouts --run-root .../baseline_v4_multidomain_squad
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis.judge_filter_sft import (
    DEFAULT_MAX_CONTEXT_CHARS,
    is_v2_training_judge_row,
    load_jsonl,
    to_sft_row,
    write_jsonl,
)
from thesis.prepare_repliqa_sft_splits import split_by_document

THESIS_ROOT = Path(__file__).resolve().parent

DOMAIN_SOURCE_PREFIX: dict[str, str] = {
    "repliqa": "repliqa/",
    "ohioline": "ohioline/",
    "quoref": "quoref/",
    "squad": "squad_v2/",
}

DEFAULT_DROP_JUDGED = (
    THESIS_ROOT / "experiments/drop/runs/drop_synthetic_full_v1/train/bedrock_judge.jsonl"
)


def _filter_val_by_prefix(val_rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [r for r in val_rows if str(r.get("source") or "").startswith(prefix)]


def export_holdouts(
    *,
    run_root: Path,
    out_dir: Path | None = None,
    drop_judged_jsonl: Path = DEFAULT_DROP_JUDGED,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> dict[str, Any]:
    splits_dir = run_root / "splits"
    val_path = splits_dir / "val.jsonl"
    if not val_path.is_file():
        raise FileNotFoundError(f"Missing {val_path}")

    holdout_dir = out_dir or (run_root / "eval_holdouts")
    holdout_dir.mkdir(parents=True, exist_ok=True)

    val_rows = load_jsonl(val_path)
    manifest_domains: dict[str, Any] = {}

    for domain, prefix in DOMAIN_SOURCE_PREFIX.items():
        rows = _filter_val_by_prefix(val_rows, prefix)
        out_path = holdout_dir / f"val_{domain}.jsonl"
        write_jsonl(out_path, rows)
        manifest_domains[domain] = {
            "source_prefix": prefix,
            "n_rows": len(rows),
            "path": str(out_path.resolve()),
            "from": "v4_val_split",
        }

    # OhioLine OOD test file (canonical 511-row holdout)
    test_ohio = splits_dir / "test_ohioline.jsonl"
    if test_ohio.is_file():
        ohioline_test = holdout_dir / "test_ohioline.jsonl"
        if not ohioline_test.exists() or ohioline_test.stat().st_mtime < test_ohio.stat().st_mtime:
            ohioline_test.write_text(test_ohio.read_text(encoding="utf-8"), encoding="utf-8")
        manifest_domains["ohioline"]["test_path"] = str(ohioline_test.resolve())
        manifest_domains["ohioline"]["test_n_rows"] = sum(
            1 for _ in ohioline_test.read_text(encoding="utf-8").splitlines() if _.strip()
        )

    # DROP: zero-shot holdout (not in v4 train); document-level val split
    drop_rows_raw = load_jsonl(drop_judged_jsonl)
    drop_usable = [r for r in drop_rows_raw if is_v2_training_judge_row(r)]
    _, drop_val_raw, drop_split = split_by_document(drop_usable, val_ratio=val_ratio, seed=seed)
    drop_val = [to_sft_row(r, max_context_chars=max_context_chars) for r in drop_val_raw]
    drop_path = holdout_dir / "val_drop.jsonl"
    write_jsonl(drop_path, drop_val)
    manifest_domains["drop"] = {
        "n_rows": len(drop_val),
        "path": str(drop_path.resolve()),
        "from": "drop_synthetic_full_v1_document_holdout",
        "judged_jsonl": str(drop_judged_jsonl.resolve()),
        "val_ratio": val_ratio,
        "seed": seed,
        "split_manifest": drop_split,
        "zero_shot": True,
    }

    manifest = {
        "schema": "judge_filter_holdouts/v1",
        "run_root": str(run_root.resolve()),
        "holdout_dir": str(holdout_dir.resolve()),
        "domains": manifest_domains,
    }
    manifest_path = holdout_dir / "holdouts_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_export_judge_filter_holdouts(ns: argparse.Namespace) -> int:
    manifest = export_holdouts(
        run_root=Path(ns.run_root).expanduser().resolve(),
        out_dir=Path(ns.out_dir).expanduser().resolve() if ns.out_dir else None,
        drop_judged_jsonl=Path(ns.drop_judged_jsonl).expanduser().resolve(),
        val_ratio=float(ns.val_ratio),
        seed=int(ns.seed),
        max_context_chars=int(ns.max_context_chars),
    )
    print(json.dumps(manifest, indent=2))
    print(f"Wrote holdouts under {manifest['holdout_dir']}", flush=True)
    return 0


def add_cli(sub: argparse._SubParsersAction) -> None:
    default_run = THESIS_ROOT / "experiments/judge_filter/runs/baseline_v4_multidomain_squad"
    p = sub.add_parser(
        "export-judge-filter-holdouts",
        help="Export per-domain val holdouts + DROP zero-shot set for judge-filter eval",
    )
    p.add_argument("--run-root", type=Path, default=default_run)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--drop-judged-jsonl", type=Path, default=DEFAULT_DROP_JUDGED)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    p.set_defaults(fn=run_export_judge_filter_holdouts)
