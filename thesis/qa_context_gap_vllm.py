"""
Per-row **context gap** via vLLM: generate an answer **without** the chunk and **with** the chunk,
then cosine similarity between embeddings of those two answers.

- ``answer_without_context``: QUESTION_ONLY_SYSTEM + answerer_question_only_block(question)
- ``answer_with_context``: ANSWERER_SYSTEM + answerer_user_block(context, question)

Requires a running vLLM OpenAI-compatible server and ``sentence-transformers`` for embeddings.

Default ``--vllm-model`` is the **base** instruct checkpoint (``paths.DEFAULT_BASE_MODEL_ID`` /
``DOMAIN_BASE_MODEL_ID``), not the merged generator LoRA — start vLLM with that same id.

gap_one_minus_cosine = 1 - cosine_similarity (after normalizing cosine to a sensible range).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import thesis.bootstrap  # noqa: F401

from paths import (
    DEFAULT_BASE_MODEL_ID,
    QA_GEN_CONCURRENCY,
    QA_GEN_MAX_NEW_TOKENS,
    generator_vllm_base_url,
)
from thesis.prompts import (
    ANSWERER_SYSTEM,
    QUESTION_ONLY_SYSTEM,
    answerer_question_only_block,
    answerer_user_block,
)


def _openai_base_url(base_url: str) -> str:
    bu = base_url.rstrip("/")
    if not bu.endswith("/v1"):
        bu = bu + "/v1"
    return bu


def _vllm_chat(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    kw: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": float(temperature),
    }
    r = client.chat.completions.create(**kw)
    return (r.choices[0].message.content or "").strip()


def _check_vllm(client: Any) -> None:
    try:
        client.models.list()
    except Exception as e:
        raise SystemExit(
            f"Cannot reach vLLM OpenAI API: {e}\n"
            "Start the server first, e.g.:\n"
            "  python -m vllm.entrypoints.openai.api_server --model <path> --host 127.0.0.1 --port 8100"
        ) from e


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# gap_one_minus_cosine tiers (partition): low < GAP_MED_LOW, medium [GAP_MED_LOW, GAP_HIGH], high > GAP_HIGH
GAP_MED_LOW = 0.2
GAP_HIGH = 0.6


def _gap_tier(gap: float) -> str:
    if gap < GAP_MED_LOW:
        return "low"
    if gap <= GAP_HIGH:
        return "medium"
    return "high"


def run_qa_context_gap_vllm(ns: argparse.Namespace) -> int:
    qa_path = Path(ns.qa_jsonl).expanduser().resolve()
    if not qa_path.is_file():
        print(f"Not found: {qa_path}", file=sys.stderr)
        return 1

    out_jsonl = (
        Path(ns.out_jsonl).expanduser().resolve()
        if ns.out_jsonl
        else qa_path.parent / f"{qa_path.stem}_context_gap.jsonl"
    )
    sum_path = (
        Path(ns.summary_json).expanduser().resolve()
        if ns.summary_json
        else out_jsonl.with_name(out_jsonl.stem + "_summary.json")
    )

    from openai import OpenAI

    base_url = _openai_base_url(ns.vllm_base_url)
    client = OpenAI(base_url=base_url, api_key="unused")
    print(f"vLLM base: {base_url}  model={ns.vllm_model!r}", flush=True)
    _check_vllm(client)

    lines = qa_path.read_text(encoding="utf-8").splitlines()
    rows_in: list[tuple[int, dict[str, Any]]] = []
    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows_in.append((line_no, json.loads(line)))
        except json.JSONDecodeError as e:
            print(f"Line {line_no}: JSON error: {e}", file=sys.stderr)

    if ns.max_rows > 0:
        rows_in = rows_in[: ns.max_rows]

    max_tok = int(ns.max_new_tokens)
    temp = float(ns.temperature)
    conc = max(1, int(ns.concurrency))

    results: list[dict[str, Any] | None] = [None] * len(rows_in)
    lock = threading.Lock()
    done = 0

    def one(idx: int, line_no: int, row: dict[str, Any]) -> None:
        nonlocal done
        q = str(row.get("question") or "").strip()
        ctx = str(row.get("context") or "").strip()
        out: dict[str, Any] = {**row}
        gblock: dict[str, Any] = {
            "vllm_model": ns.vllm_model,
            "max_new_tokens": max_tok,
            "temperature": temp,
        }
        if not q:
            gblock["error"] = "missing_question"
            gblock["answer_without_context"] = None
            gblock["answer_with_context"] = None
            out["context_gap"] = gblock
            with lock:
                results[idx] = out
                done += 1
            return
        if not ctx:
            gblock["error"] = "missing_context"
            gblock["answer_without_context"] = None
            gblock["answer_with_context"] = None
            out["context_gap"] = gblock
            with lock:
                results[idx] = out
                done += 1
            return

        messages_no = [
            {"role": "system", "content": QUESTION_ONLY_SYSTEM},
            {"role": "user", "content": answerer_question_only_block(q)},
        ]
        messages_yes = [
            {"role": "system", "content": ANSWERER_SYSTEM},
            {"role": "user", "content": answerer_user_block(ctx, q)},
        ]
        try:
            a_no = _vllm_chat(
                client=client,
                model=ns.vllm_model,
                messages=messages_no,
                max_tokens=max_tok,
                temperature=temp,
            )
        except Exception as e:
            gblock["error"] = f"vllm_without_context: {e}"
            gblock["answer_without_context"] = None
            gblock["answer_with_context"] = None
            out["context_gap"] = gblock
            with lock:
                results[idx] = out
                done += 1
            return
        try:
            a_yes = _vllm_chat(
                client=client,
                model=ns.vllm_model,
                messages=messages_yes,
                max_tokens=max_tok,
                temperature=temp,
            )
        except Exception as e:
            gblock["error"] = f"vllm_with_context: {e}"
            gblock["answer_without_context"] = a_no
            gblock["answer_with_context"] = None
            out["context_gap"] = gblock
            with lock:
                results[idx] = out
                done += 1
            return

        gblock["answer_without_context"] = a_no
        gblock["answer_with_context"] = a_yes
        out["context_gap"] = gblock
        with lock:
            results[idx] = out
            done += 1
            if done % 10 == 0 or done == len(rows_in):
                print(f"  ... vLLM {done}/{len(rows_in)} rows", flush=True)

    print(
        f"Generating (no-chunk + with-chunk): concurrency={conc}, "
        f"max_new_tokens={max_tok}, temperature={temp}, rows={len(rows_in)}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [
            pool.submit(one, i, line_no, row)
            for i, (line_no, row) in enumerate(rows_in)
        ]
        for f in as_completed(futs):
            f.result()

    # Embeddings: batch pairs where both answers exist and non-empty
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "Missing sentence-transformers; embeddings skipped.\n"
            "  pip install sentence-transformers",
            file=sys.stderr,
        )
        SentenceTransformer = None  # type: ignore

    embed_model_name = str(ns.embed_model)
    device = ns.embed_device
    if SentenceTransformer is not None:
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embed model {embed_model_name} on {device} …", flush=True)
        st_model = SentenceTransformer(embed_model_name, device=device)

    import numpy as np

    left: list[str] = []
    right: list[str] = []
    idx_map: list[int] = []
    for i, r in enumerate(results):
        if r is None:
            continue
        cg = r.get("context_gap") or {}
        if not isinstance(cg, dict) or cg.get("error"):
            continue
        a0 = cg.get("answer_without_context")
        a1 = cg.get("answer_with_context")
        if not isinstance(a0, str) or not isinstance(a1, str):
            continue
        if not a0.strip() or not a1.strip():
            cg["embed_error"] = "empty_generated_answer"
            r["context_gap"] = cg
            continue
        idx_map.append(i)
        left.append(a0)
        right.append(a1)

    if SentenceTransformer is not None and left:
        le = st_model.encode(
            left,
            batch_size=int(ns.embed_batch_size),
            show_progress_bar=len(left) > 64,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        re = st_model.encode(
            right,
            batch_size=int(ns.embed_batch_size),
            show_progress_bar=len(right) > 64,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        cos_arr = np.sum(le * re, axis=1)
        for j, row_i in enumerate(idx_map):
            r = results[row_i]
            assert r is not None
            cg = dict(r.get("context_gap") or {})
            c = float(np.clip(cos_arr[j], -1.0, 1.0))
            g = 1.0 - c
            cg["embed_model"] = embed_model_name
            cg["cosine_between_generated_answers"] = round(c, 6)
            cg["gap_one_minus_cosine"] = round(g, 6)
            cg["gap_tier"] = _gap_tier(g)
            r["context_gap"] = cg
            results[row_i] = r

    rows_out: list[dict[str, Any]] = []
    for r in results:
        if r is not None:
            rows_out.append(r)

    sum_cos = 0.0
    sum_gap = 0.0
    n_embed = 0
    n_vllm_ok = 0
    n_gap_low = 0
    n_gap_medium = 0
    n_gap_high = 0
    for r in rows_out:
        cg = r.get("context_gap") or {}
        if isinstance(cg, dict) and cg.get("answer_without_context") is not None and cg.get("answer_with_context") is not None:
            n_vllm_ok += 1
        if isinstance(cg, dict) and "cosine_between_generated_answers" in cg:
            sum_cos += float(cg["cosine_between_generated_answers"])
            sum_gap += float(cg["gap_one_minus_cosine"])
            n_embed += 1
            g = float(cg["gap_one_minus_cosine"])
            if g < GAP_MED_LOW:
                n_gap_low += 1
            elif g <= GAP_HIGH:
                n_gap_medium += 1
            else:
                n_gap_high += 1

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fp:
        for r in rows_out:
            fp.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    summary = {
        "schema": "qa_context_gap_vllm_summary/v1",
        "qa_jsonl": str(qa_path),
        "out_jsonl": str(out_jsonl),
        "vllm_base_url": ns.vllm_base_url,
        "vllm_model": ns.vllm_model,
        "settings": {
            "max_new_tokens": int(ns.max_new_tokens),
            "temperature": float(ns.temperature),
            "concurrency": int(ns.concurrency),
            "embed_model": embed_model_name if SentenceTransformer else None,
        },
        "stats": {
            "n_rows": len(rows_out),
            "n_vllm_pairs_ok": n_vllm_ok,
            "n_embedded_pairs": n_embed,
            "mean_cosine_between_generated_answers": round(sum_cos / max(1, n_embed), 6) if n_embed else None,
            "mean_gap_one_minus_cosine": round(sum_gap / max(1, n_embed), 6) if n_embed else None,
            "gap_tier_counts": {
                "low_gap_lt_0_2": n_gap_low,
                "medium_gap_0_2_to_0_6": n_gap_medium,
                "high_gap_gt_0_6": n_gap_high,
            },
        },
        "gap_tier_boundaries": {
            "low": f"gap_one_minus_cosine < {GAP_MED_LOW}",
            "medium": f"{GAP_MED_LOW} <= gap_one_minus_cosine <= {GAP_HIGH}",
            "high": f"gap_one_minus_cosine > {GAP_HIGH}",
        },
        "notes": [
            "High gap_one_minus_cosine => answers differ more in embedding space (document may have shifted the model).",
            "Use temperature=0 for more stable repeats; embeddings require non-empty both generations.",
            "gap_tier on each row: low | medium | high (see gap_tier_boundaries).",
        ],
    }
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_jsonl}", flush=True)
    print(f"Wrote {sum_path}", flush=True)
    me = summary["stats"]["mean_cosine_between_generated_answers"]
    mg = summary["stats"]["mean_gap_one_minus_cosine"]
    gt = summary["stats"]["gap_tier_counts"]
    print(f"Rows: {len(rows_out)}  vLLM pairs ok: {n_vllm_ok}  embedded: {n_embed}  mean cos={me}  mean gap={mg}", flush=True)
    print(
        f"Gap tiers:  low (<{GAP_MED_LOW})={gt['low_gap_lt_0_2']}  "
        f"medium ([{GAP_MED_LOW},{GAP_HIGH}])={gt['medium_gap_0_2_to_0_6']}  "
        f"high (>{GAP_HIGH})={gt['high_gap_gt_0_6']}",
        flush=True,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="vLLM context-gap: answer without chunk vs with chunk, then embedding cosine between answers."
    )
    p.add_argument("--qa-jsonl", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, default=None)
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument(
        "--vllm-base-url",
        type=str,
        default=generator_vllm_base_url(),
        help="OpenAI-compatible base URL (vLLM).",
    )
    p.add_argument(
        "--vllm-model",
        type=str,
        default=DEFAULT_BASE_MODEL_ID,
        help="Model id as served by vLLM (default: base instruct; override DOMAIN_BASE_MODEL_ID).",
    )
    p.add_argument("--max-new-tokens", type=int, default=QA_GEN_MAX_NEW_TOKENS)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default 0 for reproducible pairs; was QA_GEN_TEMPERATURE in other scripts).",
    )
    p.add_argument("--concurrency", type=int, default=QA_GEN_CONCURRENCY)
    p.add_argument("--max-rows", type=int, default=0, help="0 = all rows.")
    p.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)
    p.add_argument("--embed-device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--embed-batch-size", type=int, default=32)
    return p


if __name__ == "__main__":
    ns = build_arg_parser().parse_args()
    raise SystemExit(run_qa_context_gap_vllm(ns))
