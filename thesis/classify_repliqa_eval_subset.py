"""
Classify RepLiQA eval questions with **Llama 3.2 3B Instruct** (default) or optional heuristics.

Purpose: slice eval metrics by question difficulty / evidence type so you can see where
finetuning helps (e.g. factual lookup with answer in paragraph → low expected gain).

See category definitions:
  thesis/experiments/repliqa/runs/repliqa_train_0-3/eval/EVAL_QUESTION_CATEGORIES.md

Usage (from finetuning/):
  source thesis/scripts/source_bedrock_env.sh
  python -m thesis.cli classify-repliqa-eval-subset --backend bedrock
  python -m thesis.cli classify-repliqa-eval-subset --backend llm   # GPU, local 3B
  python -m thesis.cli classify-repliqa-eval-subset --backend heuristic
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# At least one of these keys must be present (do not use parse_judge_json — it requires "overall")
_CLASSIFIER_JSON_MARKERS = ("answer_evidence", "finetuning_expected_gain", "question_type")

# Allowed enum values (validated after LLM parse)
ALLOWED = {
    "question_type": {
        "what", "who", "when", "where", "why", "how", "which", "yes_no", "other",
    },
    "answer_evidence": {
        "explicit_span",
        "paraphrase",
        "multi_span",
        "light_inference",
        "heavy_inference",
    },
    "document_necessity": {"required", "helpful", "not_required"},
    "answer_form": {"entity", "date_or_number", "short_phrase", "sentence"},
    "finetuning_expected_gain": {"low", "medium", "high"},
}

CLASSIFIER_SYSTEM = (
    "You classify document-grounded QA examples for a research evaluation. "
    "Read the context, question, and gold reference answer. "
    "Respond with a single JSON object only — no markdown, no extra text."
)

CLASSIFIER_USER_TEMPLATE = """Classify this QA example for offline evaluation analysis.

Context:
{context}

Question:
{question}

Gold reference answer (human benchmark label — use only to judge evidence type, not to score models):
{gold}

Assign these fields (use ONLY the allowed enum values):

question_type — surface form of the question:
  what | who | when | where | why | how | which | yes_no | other

answer_evidence — how the gold answer relates to the context:
  explicit_span — answer or near-verbatim phrase is clearly stated in context (lookup/copy)
  paraphrase — facts are in context but must be rephrased
  multi_span — must combine two or more non-adjacent parts of the context
  light_inference — small synthesis from facts that are stated
  heavy_inference — answer not fully explicit; non-trivial reasoning

document_necessity — is this specific document needed?
  required | helpful | not_required

answer_form — shape of the gold answer:
  entity | date_or_number | short_phrase | sentence

finetuning_expected_gain — expected benefit of Q/A finetuning vs base model with same context:
  low — factual lookup; answer explicit in paragraph; base+context likely enough
  medium — paraphrase or light synthesis or document-specific wording
  high — multi-span, heavy inference, or strong benefit from trained Q/A style

brief_reason — one sentence justification.

JSON only:
{{"question_type": "...", "answer_evidence": "...", "document_necessity": "...", "answer_form": "...", "finetuning_expected_gain": "...", "brief_reason": "..."}}
"""


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i} {e}") from e
    return rows


def load_train_document_ids(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    return {
        (r.get("document_id") or "").strip()
        for r in load_jsonl(path)
        if (r.get("document_id") or "").strip()
    }


def parse_question_index_in_doc(row: dict[str, Any]) -> int | None:
    for key in ("chunk_id", "eval_id"):
        val = (row.get(key) or "").strip()
        m = re.search(r"-q(\d+)$", val, re.I)
        if m:
            return int(m.group(1))
    return None


def metadata_categories(row: dict[str, Any], train_doc_ids: set[str]) -> dict[str, Any]:
    doc_id = (row.get("document_id") or "").strip()
    return {
        "document_topic": (row.get("document_topic") or "unknown").strip() or "unknown",
        "repliqa_split": (row.get("repliqa_split") or "unknown").strip() or "unknown",
        "question_index_in_doc": parse_question_index_in_doc(row),
        "in_synthetic_train_document": bool(doc_id and doc_id in train_doc_ids),
    }


def parse_classifier_json(raw: str) -> dict[str, Any] | None:
    """Parse Llama classifier output (not the Haiku judge schema)."""
    s = raw.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", s, re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and any(k in obj for k in _CLASSIFIER_JSON_MARKERS):
            return obj
    except json.JSONDecodeError:
        pass
    pattern = r"\{[\s\S]*?\"(?:answer_evidence|finetuning_expected_gain)\"[\s\S]*?\}"
    m = re.search(pattern, s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def _normalize_enum(value: Any, allowed: set[str], default: str) -> str:
    s = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if s in allowed:
        return s
    # fuzzy: strip suffixes
    for a in allowed:
        if s == a or s.endswith(a):
            return a
    return default


def validate_llm_categories(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_type": _normalize_enum(
            parsed.get("question_type"), ALLOWED["question_type"], "other"
        ),
        "answer_evidence": _normalize_enum(
            parsed.get("answer_evidence"), ALLOWED["answer_evidence"], "paraphrase"
        ),
        "document_necessity": _normalize_enum(
            parsed.get("document_necessity"), ALLOWED["document_necessity"], "required"
        ),
        "answer_form": _normalize_enum(
            parsed.get("answer_form"), ALLOWED["answer_form"], "short_phrase"
        ),
        "finetuning_expected_gain": _normalize_enum(
            parsed.get("finetuning_expected_gain"),
            ALLOWED["finetuning_expected_gain"],
            "medium",
        ),
        "brief_reason": str(parsed.get("brief_reason", ""))[:500],
    }


def build_classifier_user_message(
    *,
    context: str,
    question: str,
    gold: str,
    max_context_chars: int,
) -> str:
    ctx = context[:max_context_chars] if max_context_chars > 0 else context
    return CLASSIFIER_USER_TEMPLATE.format(context=ctx, question=question, gold=gold)


def classify_row_with_bedrock(
    client: Any,
    row: dict[str, Any],
    *,
    model_id: str,
    max_context_chars: int,
    max_tokens: int,
) -> dict[str, Any]:
    from thesis.bedrock_judge_qa_score import _invoke_bedrock_claude

    ctx = (row.get("context") or "").strip()
    q = (row.get("question") or "").strip()
    gold = (row.get("gold") or row.get("answer") or "").strip()
    if not ctx or not q or not gold:
        return {
            "classifier_error": "missing_context_question_or_gold",
            "classifier_backend": "bedrock",
            "classifier_model": model_id,
        }

    user = build_classifier_user_message(
        context=ctx, question=q, gold=gold, max_context_chars=max_context_chars
    )
    try:
        raw = _invoke_bedrock_claude(
            client,
            model_id=model_id,
            user_message=user,
            max_tokens=max_tokens,
            temperature=0.0,
            system=CLASSIFIER_SYSTEM,
        )
    except Exception as e:
        return {
            "classifier_error": f"api_error:{e}",
            "classifier_backend": "bedrock",
            "classifier_model": model_id,
        }

    parsed = parse_classifier_json(raw)
    if not parsed:
        return {
            "classifier_error": "parse_error",
            "classifier_backend": "bedrock",
            "classifier_model": model_id,
            "raw_preview": raw[:400],
        }
    llm_cats = validate_llm_categories(parsed)
    return {
        **llm_cats,
        "classifier_backend": "bedrock",
        "classifier_model": model_id,
    }


def classify_row_with_llm(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    model_id: str,
    max_context_chars: int,
    max_new_tokens: int,
    use_bf16: bool,
) -> dict[str, Any]:
    import torch
    from transformers import GenerationConfig

    ctx = (row.get("context") or "").strip()
    q = (row.get("question") or "").strip()
    gold = (row.get("gold") or row.get("answer") or "").strip()
    if not ctx or not q or not gold:
        return {
            "classifier_error": "missing_context_question_or_gold",
            "classifier_backend": "llm",
            "classifier_model": model_id,
        }

    user = build_classifier_user_message(
        context=ctx, question=q, gold=gold, max_context_chars=max_context_chars
    )
    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM},
        {"role": "user", "content": user},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )
    device = next(model.parameters()).device
    if isinstance(inputs, torch.Tensor):
        gen_in = {"input_ids": inputs.to(device)}
    else:
        gen_in = {k: v.to(device) for k, v in dict(inputs).items()}
    if "attention_mask" not in gen_in:
        gen_in["attention_mask"] = torch.ones_like(gen_in["input_ids"], dtype=torch.long, device=device)

    input_len = gen_in["input_ids"].shape[-1]
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        out = model.generate(**gen_in, generation_config=gen_cfg)
    raw = tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()
    parsed = parse_classifier_json(raw)
    if not parsed:
        return {
            "classifier_error": "parse_error",
            "classifier_backend": "llm",
            "classifier_model": model_id,
            "raw_preview": raw[:400],
        }
    llm_cats = validate_llm_categories(parsed)
    return {
        **llm_cats,
        "classifier_backend": "llm",
        "classifier_model": model_id,
    }


def _classify_heuristic_question_type(question: str) -> str:
    q = (question or "").strip()
    rules = [
        ("how_many", r"^how many\b"),
        ("how_much", r"^how much\b"),
        ("what", r"^what\b"),
        ("who", r"^(who|whom|whose)\b"),
        ("when", r"^when\b"),
        ("where", r"^where\b"),
        ("why", r"^why\b"),
        ("how", r"^how\b"),
        ("which", r"^which\b"),
        ("yes_no", r"^(do|does|did|is|are|was|were|can|could|will|would|has|have)\b"),
    ]
    for label, pat in rules:
        if re.search(pat, q, re.I):
            return label if label in ALLOWED["question_type"] else "other"
    return "other"


def classify_row_heuristic(row: dict[str, Any]) -> dict[str, Any]:
    """Fallback only — does not set answer_evidence or finetuning_expected_gain well."""
    gold = (row.get("gold") or row.get("answer") or "").strip()
    ctx = (row.get("context") or "").strip()
    q = (row.get("question") or "").strip()
    # crude: gold substring in context → explicit_span
    gold_norm = re.sub(r"\s+", " ", gold.lower())
    ctx_norm = re.sub(r"\s+", " ", ctx.lower())
    if gold_norm and gold_norm in ctx_norm:
        evidence = "explicit_span"
        gain = "low"
    else:
        evidence = "paraphrase"
        gain = "medium"
    n_tok = len(re.findall(r"\b\w+\b", gold.lower()))
    if n_tok <= 3:
        form = "entity"
    elif re.search(r"\d|20\d{2}|january|february", gold, re.I):
        form = "date_or_number"
    elif n_tok <= 12:
        form = "short_phrase"
    else:
        form = "sentence"

    return {
        "question_type": _classify_heuristic_question_type(q),
        "answer_evidence": evidence,
        "document_necessity": "required",
        "answer_form": form,
        "finetuning_expected_gain": gain,
        "brief_reason": "heuristic_fallback",
        "classifier_backend": "heuristic",
        "classifier_model": None,
    }


def cross_tab(rows: list[dict[str, Any]], key_a: str, key_b: str) -> dict[str, dict[str, int]]:
    tab: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        cat = row.get("categories") or {}
        tab[str(cat.get(key_a, "unknown"))][str(cat.get(key_b, "unknown"))] += 1
    return {a: dict(b) for a, b in tab.items()}


def run_classify_repliqa_eval_subset(ns: argparse.Namespace) -> int:
    eval_dir = Path(ns.eval_dir).expanduser().resolve()
    run_root = Path(ns.run_root).expanduser().resolve()
    in_path = Path(ns.input_jsonl).expanduser().resolve() if ns.input_jsonl else eval_dir / ns.input_name
    if not in_path.is_file():
        print(f"Not found: {in_path}", file=sys.stderr)
        return 1

    backend = str(ns.backend).lower()
    default_out_names = {
        "llm": "eval_subset_2000_classified.jsonl",
        "bedrock": "eval_subset_2000_classified_bedrock.jsonl",
        "heuristic": "eval_subset_2000_classified_heuristic.jsonl",
    }
    if ns.output_jsonl:
        out_path = Path(ns.output_jsonl).expanduser().resolve()
    elif ns.output_name != "eval_subset_2000_classified.jsonl":
        out_path = eval_dir / ns.output_name
    else:
        out_path = eval_dir / default_out_names.get(backend, ns.output_name)

    summary_path = (
        Path(ns.summary_json).expanduser().resolve()
        if ns.summary_json
        else eval_dir / ns.summary_name
    )

    train_doc_path = (
        Path(ns.train_documents_jsonl).expanduser().resolve()
        if ns.train_documents_jsonl
        else run_root / "train/documents_unique.jsonl"
    )
    train_doc_ids = load_train_document_ids(train_doc_path)

    rows = load_jsonl(in_path)
    if int(ns.max_rows) > 0:
        rows = rows[: int(ns.max_rows)]
    if not rows:
        print("No rows.", file=sys.stderr)
        return 1

    wall0 = time.perf_counter()
    model = tokenizer = client = None
    model_id = str(ns.model or DEFAULT_MODEL_ID)

    if backend == "bedrock":
        from thesis.bedrock_judge_qa_score import (
            DEFAULT_BEDROCK_MODEL_ID,
            _bedrock_client,
            _check_aws_env,
        )

        if not str(ns.model or "").strip() or ns.model == DEFAULT_MODEL_ID:
            model_id = os.environ.get("BEDROCK_JUDGE_MODEL_ID") or DEFAULT_BEDROCK_MODEL_ID
        region = (
            os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or ""
        ).strip()
        _check_aws_env(region)
        if (os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or "").strip() and (
            os.environ.get("AWS_ACCESS_KEY_ID") or ""
        ).strip():
            os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
        client = _bedrock_client(region)
        conc = max(1, int(ns.concurrency))
        delay = float(ns.request_delay_s)
        max_tok = int(ns.max_new_tokens)
        print(
            f"Bedrock classifier: region={region} model={model_id} rows={len(rows)} "
            f"concurrency={conc}",
            flush=True,
        )

    if backend == "llm":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise SystemExit(
                "Missing deps for --backend llm. Install: pip install -r requirements_sft.txt"
            ) from e

        print(f"Loading classifier model {model_id} …", flush=True)
        t_load = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if ns.bf16 and torch.cuda.is_available() else None
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )
        model.eval()
        load_s = time.perf_counter() - t_load
        print(f"Model ready in {load_s:.1f}s", flush=True)

    rows_out: list[dict[str, Any] | None] = [None] * len(rows)
    n_err = 0
    lock = threading.Lock()

    if backend == "bedrock":
        assert client is not None

        def job(i: int, row: dict[str, Any]) -> None:
            nonlocal n_err
            meta = metadata_categories(row, train_doc_ids)
            if delay > 0:
                time.sleep(delay * (i % conc) * 0.05)
            llm_part = classify_row_with_bedrock(
                client,
                row,
                model_id=model_id,
                max_context_chars=int(ns.max_context_chars),
                max_tokens=max_tok,
            )
            with lock:
                if llm_part.get("classifier_error"):
                    n_err += 1
                rows_out[i] = {**row, "categories": {**meta, **llm_part}}
                done = sum(1 for r in rows_out if r is not None)
                if done % 20 == 0 or done == len(rows):
                    print(f"  classified {done}/{len(rows)}", flush=True)

        with ThreadPoolExecutor(max_workers=conc) as pool:
            futs = [pool.submit(job, i, r) for i, r in enumerate(rows)]
            for f in as_completed(futs):
                f.result()
        rows_out = [r for r in rows_out if r is not None]
    else:
        for i, row in enumerate(rows):
            meta = metadata_categories(row, train_doc_ids)
            if backend == "llm":
                assert model is not None and tokenizer is not None
                llm_part = classify_row_with_llm(
                    model,
                    tokenizer,
                    row,
                    model_id=model_id,
                    max_context_chars=int(ns.max_context_chars),
                    max_new_tokens=int(ns.max_new_tokens),
                    use_bf16=bool(ns.bf16),
                )
                if llm_part.get("classifier_error"):
                    n_err += 1
            else:
                llm_part = classify_row_heuristic(row)

            rows_out.append({**row, "categories": {**meta, **llm_part}})
            if backend == "llm" and ((i + 1) % 20 == 0 or i + 1 == len(rows)):
                print(f"  classified {i + 1}/{len(rows)}", flush=True)

    wall_s = time.perf_counter() - wall0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        for r in rows_out:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _count_field(field: str) -> dict[str, int]:
        c: Counter[str] = Counter()
        for r in rows_out:
            v = (r.get("categories") or {}).get(field)
            if v is not None and not str(v).startswith("classifier_"):
                c[str(v)] += 1
            elif (r.get("categories") or {}).get("classifier_error"):
                c["__error__"] += 1
        return dict(c)

    summary = {
        "schema": "repliqa_eval_classification_summary/v2",
        "created_utc": _utc_iso(),
        "classifier_backend": backend,
        "classifier_model": model_id if backend in ("llm", "bedrock") else None,
        "input_jsonl": str(in_path),
        "output_jsonl": str(out_path),
        "n_rows": len(rows_out),
        "n_classifier_errors": n_err,
        "timing": {"total_wall_s": round(wall_s, 3)},
        "counts": {
            "question_type": _count_field("question_type"),
            "answer_evidence": _count_field("answer_evidence"),
            "document_necessity": _count_field("document_necessity"),
            "answer_form": _count_field("answer_form"),
            "finetuning_expected_gain": _count_field("finetuning_expected_gain"),
            "document_topic": _count_field("document_topic"),
            "repliqa_split": _count_field("repliqa_split"),
        },
        "cross_tabs": {
            "finetuning_expected_gain_x_answer_evidence": cross_tab(
                rows_out, "finetuning_expected_gain", "answer_evidence"
            ),
            "finetuning_expected_gain_x_document_topic": cross_tab(
                rows_out, "finetuning_expected_gain", "document_topic"
            ),
        },
        "category_doc": str(eval_dir / "EVAL_QUESTION_CATEGORIES.md"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_path} ({len(rows_out)} rows, backend={backend})", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    if backend in ("llm", "bedrock"):
        print(f"Classifier errors: {n_err}", flush=True)
    print(
        "finetuning_expected_gain:",
        summary["counts"].get("finetuning_expected_gain"),
        flush=True,
    )
    return 0 if len(rows_out) > n_err else 1


def build_arg_parser() -> argparse.ArgumentParser:
    run_root = Path(__file__).resolve().parent / "experiments/repliqa/runs/repliqa_train_0-3"
    eval_dir = run_root / "eval"
    p = argparse.ArgumentParser(
        description="Classify eval questions with Llama 3.2 (default) or heuristics."
    )
    p.add_argument("--run-root", type=Path, default=run_root)
    p.add_argument("--eval-dir", type=Path, default=eval_dir)
    p.add_argument("--input-jsonl", type=Path, default=None)
    p.add_argument("--input-name", type=str, default="eval_subset_2000.jsonl")
    p.add_argument("--output-jsonl", type=Path, default=None)
    p.add_argument(
        "--output-name",
        type=str,
        default="eval_subset_2000_classified.jsonl",
    )
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--summary-name", type=str, default="eval_classification_summary.json")
    p.add_argument("--train-documents-jsonl", type=Path, default=None)
    p.add_argument(
        "--backend",
        type=str,
        choices=("bedrock", "llm", "heuristic"),
        default="bedrock",
        help="bedrock = Haiku via AWS (recommended); llm = local 3B GPU; heuristic = rules",
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    p.add_argument("--max-context-chars", type=int, default=12000)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--concurrency", type=int, default=4, help="bedrock only")
    p.add_argument("--request-delay-s", type=float, default=0.1, help="bedrock throttle")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", action="store_true")
    return p


if __name__ == "__main__":
    ns = build_arg_parser().parse_args()
    if ns.no_bf16:
        ns.bf16 = False
    raise SystemExit(run_classify_repliqa_eval_subset(ns))
