"""
NLI-based faithfulness scoring for QA JSONL (context = premise, answer = hypothesis).

Uses a small MNLI model (default: MoritzLaurer/DeBERTa-v3-base-mnli on Hugging Face).

Long **context**: sliding windows over premise tokens, then aggregate probs across windows.
Long **answer**: token-truncate hypothesis to ``max_hypothesis_tokens`` (prefix kept).

Run: python -m thesis.cli qa-nli-score --qa-jsonl ...
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli"


def _mnli_label_indices(model) -> tuple[int, int, int]:
    """Map contradiction / neutral / entailment to logit indices from model.config."""
    lid = getattr(model.config, "label2id", None) or {}
    for a, b, c in (
        ("CONTRADICTION", "NEUTRAL", "ENTAILMENT"),
        ("contradiction", "neutral", "entailment"),
    ):
        if a in lid and b in lid and c in lid:
            return int(lid[a]), int(lid[b]), int(lid[c])
    return (0, 1, 2)


@dataclass
class WindowScore:
    p_contradiction: float
    p_neutral: float
    p_entailment: float
    n_windows: int
    answer_truncated: bool


def _softmax_three(logits: torch.Tensor, i_con: int, i_neu: int, i_ent: int) -> tuple[float, float, float]:
    # logits: [num_labels] or [1, num_labels] — softmax over classes, do NOT index [0] again (that was a bug).
    if logits.dim() == 2:
        logits = logits[0]
    p = torch.nn.functional.softmax(logits, dim=-1)
    return float(p[i_con].item()), float(p[i_neu].item()), float(p[i_ent].item())


def _pair_token_length(tokenizer, premise_text: str, hypothesis_text: str) -> int:
    """Token length of premise+hypothesis pair without truncation (for chunk sizing)."""
    enc = tokenizer(
        premise_text,
        hypothesis_text,
        add_special_tokens=True,
        truncation=False,
    )
    return len(enc["input_ids"])


def _forward_pair(
    tokenizer,
    model,
    device: torch.device,
    premise_text: str,
    hypothesis_text: str,
    max_length: int,
    *,
    i_con: int,
    i_neu: int,
    i_ent: int,
) -> tuple[float, float, float]:
    enc = tokenizer(
        premise_text,
        hypothesis_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc)
    logits = out.logits[0]
    return _softmax_three(logits, i_con, i_neu, i_ent)


def score_context_answer(
    *,
    tokenizer,
    model,
    device: torch.device,
    context: str,
    answer: str,
    max_length: int = 512,
    stride: int = 128,
    max_hypothesis_tokens: int = 384,
    aggregate: str = "max_entail",
    label_indices: tuple[int, int, int] = (0, 1, 2),
) -> tuple[dict[str, Any], WindowScore]:
    """
    aggregate:
      - max_entail: max p(entail), max p(contradiction), mean p(neutral) across windows
      - mean_probs: mean of all three probs across windows
      - max_contradict: emphasize max contradiction (max con, mean neu, max entail)
    """
    context = (context or "").strip()
    answer = (answer or "").strip()
    if not context or not answer:
        return {"error": "missing_context_or_answer"}, WindowScore(0.0, 0.0, 0.0, 0, False)

    hyp_ids = tokenizer.encode(answer, add_special_tokens=False)
    answer_truncated = False
    if len(hyp_ids) > max_hypothesis_tokens:
        hyp_ids = hyp_ids[:max_hypothesis_tokens]
        answer_truncated = True
    hyp_text = tokenizer.decode(hyp_ids, skip_special_tokens=True)

    prem_ids = tokenizer.encode(context, add_special_tokens=False)

    # Largest premise span that fits with hypothesis in one window (no silent truncation).
    def premise_fits(n_prem_tokens: int) -> bool:
        if n_prem_tokens <= 0:
            return False
        ptxt = tokenizer.decode(prem_ids[:n_prem_tokens], skip_special_tokens=True)
        return _pair_token_length(tokenizer, ptxt, hyp_text) <= max_length

    lo, hi = 1, len(prem_ids)
    chunk_cap = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if premise_fits(mid):
            chunk_cap = mid
            lo = mid + 1
        else:
            hi = mid - 1

    _guard = 0
    while chunk_cap < 32 and len(hyp_ids) > 32 and _guard < 8:
        _guard += 1
        hyp_ids = hyp_ids[: max(32, len(hyp_ids) // 2)]
        hyp_text = tokenizer.decode(hyp_ids, skip_special_tokens=True)
        answer_truncated = True
        lo, hi = 1, len(prem_ids)
        chunk_cap = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if premise_fits(mid):
                chunk_cap = mid
                lo = mid + 1
            else:
                hi = mid - 1

    all_probs: list[tuple[float, float, float]] = []

    i_con, i_neu, i_ent = label_indices
    if len(prem_ids) <= chunk_cap:
        ptxt = tokenizer.decode(prem_ids, skip_special_tokens=True)
        pc, pn, pe = _forward_pair(
            tokenizer, model, device, ptxt, hyp_text, max_length, i_con=i_con, i_neu=i_neu, i_ent=i_ent
        )
        all_probs.append((pc, pn, pe))
    else:
        start = 0
        while start < len(prem_ids):
            end = min(start + chunk_cap, len(prem_ids))
            chunk = prem_ids[start:end]
            ptxt = tokenizer.decode(chunk, skip_special_tokens=True)
            pc, pn, pe = _forward_pair(
                tokenizer, model, device, ptxt, hyp_text, max_length, i_con=i_con, i_neu=i_neu, i_ent=i_ent
            )
            all_probs.append((pc, pn, pe))
            if end >= len(prem_ids):
                break
            step = max(1, chunk_cap - stride)
            start += step

    n_win = len(all_probs)
    pcs = [x[0] for x in all_probs]
    pns = [x[1] for x in all_probs]
    pes = [x[2] for x in all_probs]

    if aggregate == "mean_probs":
        pc = sum(pcs) / n_win
        pn = sum(pns) / n_win
        pe = sum(pes) / n_win
    elif aggregate == "max_contradict":
        pc = max(pcs)
        pn = sum(pns) / n_win
        pe = max(pes)
    else:  # max_entail
        pe = max(pes)
        pc = max(pcs)
        pn = sum(pns) / n_win

    probs_vec = [pc, pn, pe]
    pred_i = max(range(3), key=lambda i: probs_vec[i])
    labels = ["contradiction", "neutral", "entailment"]
    pred_label = labels[pred_i]

    detail = {
        "p_contradiction": pc,
        "p_neutral": pn,
        "p_entailment": pe,
        "pred_label": pred_label,
        "n_windows": n_win,
        "aggregate": aggregate,
        "answer_truncated": answer_truncated,
        "max_hypothesis_tokens": max_hypothesis_tokens,
        "premise_tokens": len(prem_ids),
        "premise_chunk_cap_tokens": chunk_cap,
        "stride": stride,
        "sliding_window": n_win > 1,
    }
    return detail, WindowScore(pc, pn, pe, n_win, answer_truncated)


def run_qa_nli_score(ns: argparse.Namespace) -> int:
    qa_path = Path(ns.qa_jsonl).expanduser().resolve()
    if not qa_path.is_file():
        print(f"Not found: {qa_path}", file=sys.stderr)
        return 1

    if ns.out_jsonl:
        out_jsonl = Path(ns.out_jsonl).expanduser().resolve()
    elif getattr(ns, "sliding_window", False):
        out_jsonl = qa_path.parent / f"{qa_path.stem}_nli_sliding_window.jsonl"
    else:
        out_jsonl = qa_path.parent / f"{qa_path.stem}_nli.jsonl"
    sum_path = (
        Path(ns.summary_json).expanduser().resolve()
        if ns.summary_json
        else out_jsonl.with_name(out_jsonl.stem + "_summary.json")
    )

    device_s = ns.device
    if device_s == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_s)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"Loading {ns.model} on {device} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ns.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(ns.model)
    model.eval()
    model.to(device)
    label_indices = _mnli_label_indices(model)

    aggregate = ns.aggregate
    rows_out: list[dict[str, Any]] = []
    stats = {
        "n_rows": 0,
        "n_errors": 0,
        "sum_p_ent": 0.0,
        "sum_p_con": 0.0,
        "pred_counts": {"entailment": 0, "neutral": 0, "contradiction": 0},
        "high_contradiction_ge_0_5": 0,
        "low_entailment_le_0_25": 0,
        "answer_truncated_count": 0,
        "total_windows": 0,
        "rows_multi_window": 0,
        "max_n_windows": 0,
    }

    lines = qa_path.read_text(encoding="utf-8").splitlines()
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[misc, assignment]

    row_iter = enumerate(lines, 1)
    if tqdm is not None:
        row_iter = tqdm(row_iter, total=len(lines), desc="NLI scoring", unit="row")

    for line_no, line in row_iter:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Line {line_no}: JSON error: {e}", file=sys.stderr)
            stats["n_errors"] += 1
            continue

        ctx = row.get("context", "")
        ans = row.get("answer", "")
        nli, _ = score_context_answer(
            tokenizer=tokenizer,
            model=model,
            device=device,
            context=str(ctx),
            answer=str(ans),
            max_length=int(ns.max_length),
            stride=int(ns.stride),
            max_hypothesis_tokens=int(ns.max_hypothesis_tokens),
            aggregate=aggregate,
            label_indices=label_indices,
        )

        out_row = {**row, "nli": nli}
        rows_out.append(out_row)
        stats["n_rows"] += 1

        if "error" in nli:
            stats["n_errors"] += 1
            continue

        pe = float(nli["p_entailment"])
        pc = float(nli["p_contradiction"])
        stats["sum_p_ent"] += pe
        stats["sum_p_con"] += pc
        pl = nli.get("pred_label", "neutral")
        if pl in stats["pred_counts"]:
            stats["pred_counts"][pl] += 1
        if pc >= 0.5:
            stats["high_contradiction_ge_0_5"] += 1
        if pe <= 0.25:
            stats["low_entailment_le_0_25"] += 1
        if nli.get("answer_truncated"):
            stats["answer_truncated_count"] += 1
        n_win = int(nli.get("n_windows", 0))
        stats["total_windows"] += n_win
        if n_win > 1:
            stats["rows_multi_window"] += 1
        stats["max_n_windows"] = max(stats["max_n_windows"], n_win)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fp:
        for r in rows_out:
            fp.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    n_ok = stats["n_rows"] - stats["n_errors"]
    mean_windows = round(stats["total_windows"] / max(1, n_ok), 3)
    summary = {
        "schema": "qa_nli_benchmark_summary/v2_sliding_window",
        "model": ns.model,
        "qa_jsonl": str(qa_path),
        "out_jsonl": str(out_jsonl),
        "device": str(device),
        "settings": {
            "max_length": int(ns.max_length),
            "stride": int(ns.stride),
            "max_hypothesis_tokens": int(ns.max_hypothesis_tokens),
            "aggregate": aggregate,
            "sliding_window_output": bool(getattr(ns, "sliding_window", False)),
            "premise_fit_without_truncation": True,
        },
        "stats": {
            **stats,
            "mean_p_entailment": round(stats["sum_p_ent"] / max(1, n_ok), 6),
            "mean_p_contradiction": round(stats["sum_p_con"] / max(1, n_ok), 6),
            "mean_n_windows_per_row": mean_windows,
        },
        "notes": [
            "Long context: sliding windows over premise tokens; chunk_cap = max premise tokens per window (pair length <= max_length, no truncation in fit check).",
            "Default aggregate=max_entail (max p(entail), max p(con), mean p(neu) across windows).",
            "Long answer: truncated to max_hypothesis_tokens from the start of the encoded answer.",
            "NLI scores are heuristic; use alongside manual review for borderline rows.",
        ],
    }
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_jsonl}", flush=True)
    print(f"Wrote {sum_path}", flush=True)
    print(
        f"Rows: {stats['n_rows']}  mean p(entail)={summary['stats']['mean_p_entailment']}  "
        f"p(con)≥0.5: {stats['high_contradiction_ge_0_5']}  "
        f"mean windows/row={mean_windows}  multi-window rows={stats['rows_multi_window']}",
        flush=True,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score QA JSONL with NLI (context vs answer).")
    p.add_argument("--qa-jsonl", type=Path, required=True, help="Input JSONL with context + answer fields.")
    p.add_argument("--out-jsonl", type=Path, default=None, help="Output JSONL (default: <stem>_nli.jsonl next to input).")
    p.add_argument("--summary-json", type=Path, default=None, help="Summary JSON path.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--stride", type=int, default=128, help="Premise stride between sliding windows (token overlap ≈ chunk_cap - stride).")
    p.add_argument(
        "--max-hypothesis-tokens",
        type=int,
        default=384,
        help="Truncate encoded answer length beyond this (prefix tokens kept).",
    )
    p.add_argument(
        "--aggregate",
        type=str,
        default="max_entail",
        choices=("max_entail", "mean_probs", "max_contradict"),
        help="How to combine sliding-window scores.",
    )
    p.add_argument(
        "--sliding-window",
        action="store_true",
        help="Write outputs to <stem>_nli_sliding_window.jsonl (uses fixed multi-window scoring).",
    )
    return p


if __name__ == "__main__":
    ns = build_arg_parser().parse_args()
    raise SystemExit(run_qa_nli_score(ns))
