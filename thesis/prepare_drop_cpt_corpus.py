"""
Build passage-only corpus for DROP domain CPT (Stage 1).

Uses **train + validation** QA JSONL for passages (full knowledge base).
QA questions in validation.jsonl remain held out for LoRA training / eval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis.prepare_qa_cpt_corpus import prepare_qa_cpt_corpus


def run_prepare(ns: argparse.Namespace) -> int:
    train_jsonl = Path(ns.train_jsonl).expanduser().resolve()
    out_dir = Path(ns.out_dir).expanduser().resolve()

    qa_paths = [train_jsonl]
    if ns.eval_jsonl:
        qa_paths.append(Path(ns.eval_jsonl).expanduser().resolve())
    elif ns.no_eval_jsonl:
        pass
    else:
        default_eval = train_jsonl.parent / "validation.jsonl"
        if default_eval.is_file():
            qa_paths.append(default_eval)

    monitor_ratio = float(ns.cpt_monitor_val_ratio)
    if hasattr(ns, "val_ratio") and ns.val_ratio is not None and ns.cpt_monitor_val_ratio == 0.0:
        # Back-compat: old --val-ratio meant CPT monitor split
        if float(ns.val_ratio) > 0:
            monitor_ratio = float(ns.val_ratio)

    manifest = prepare_qa_cpt_corpus(
        qa_jsonl_paths=qa_paths,
        out_dir=out_dir,
        dataset_tag="drop",
        cpt_monitor_val_ratio=monitor_ratio,
        seed=int(ns.seed),
        min_context_chars=int(ns.min_context_chars),
        max_passages=int(ns.max_passages),
    )
    print(json.dumps(manifest, indent=2))
    print(f"Wrote CPT corpus under {manifest['out_dir']}", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare DROP passage corpus for domain CPT (full KB)")
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument(
        "--eval-jsonl",
        type=Path,
        default=None,
        help="Validation QA JSONL (default: <train-dir>/validation.jsonl if present)",
    )
    p.add_argument("--no-eval-jsonl", action="store_true", help="Train passages only (legacy)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--cpt-monitor-val-ratio",
        type=float,
        default=0.0,
        help="Passage fraction for CPT eval loss only (0 = all passages in CPT train)",
    )
    p.add_argument("--val-ratio", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-context-chars", type=int, default=40)
    p.add_argument("--max-passages", type=int, default=0)
    return p


if __name__ == "__main__":
    raise SystemExit(run_prepare(build_arg_parser().parse_args()))
