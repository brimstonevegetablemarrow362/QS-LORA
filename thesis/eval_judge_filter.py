"""
Evaluate distilled training-filter judge vs teacher on OhioLine OOD test.

  python -m thesis.cli eval-judge-filter \\
    --adapter-dir thesis/experiments/judge_filter/runs/baseline_v1/model \\
    --test-jsonl thesis/experiments/judge_filter/runs/baseline_v1/splits/test_ohioline.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thesis.judge_filter_sft import (
    DEFAULT_MAX_CONTEXT_CHARS,
    load_jsonl,
    teacher_quality_tier,
    write_jsonl,
)
from thesis.qa_judge_common import (
    JUDGE_SYSTEM,
    build_judge_user_message,
    normalize_judge_block,
    parse_judge_json,
    quality_tier_from_scores,
)

PROVIDER = "local"
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    if len(labels_a) != len(labels_b) or not labels_a:
        return None
    classes = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)
    conf: dict[str, dict[str, int]] = {a: {b: 0 for b in classes} for a in classes}
    for a, b in zip(labels_a, labels_b):
        conf[a][b] += 1
    po = sum(conf[c][c] for c in classes) / n
    row_m = {a: sum(conf[a].values()) / n for a in classes}
    col_m = {b: sum(conf[a][b] for a in classes) / n for b in classes}
    pe = sum(row_m[c] * col_m[c] for c in classes)
    if math.isclose(1.0 - pe, 0.0):
        return 1.0 if math.isclose(po, 1.0) else 0.0
    return (po - pe) / (1.0 - pe)


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    n = len(x)

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg_rank
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den_x = math.sqrt(sum((v - mx) ** 2 for v in rx))
    den_y = math.sqrt(sum((v - my) ** 2 for v in ry))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def load_judge_model(
    *,
    base_model: str,
    adapter_dir: Path | None,
    use_bf16: bool,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir or base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer


def predict_judge_block(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_context_chars: int,
    max_new_tokens: int,
    use_bf16: bool,
) -> dict[str, Any]:
    import torch
    from transformers import GenerationConfig

    ctx = str(row.get("context") or "")
    q = str(row.get("question") or "")
    ans = str(row.get("answer") or "")
    user = build_judge_user_message(
        context=ctx,
        question=q,
        answer=ans,
        max_context_chars=max_context_chars,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
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
    parsed = parse_judge_json(raw)
    try:
        return normalize_judge_block(
            provider=PROVIDER,
            model=str(getattr(model, "name_or_path", DEFAULT_MODEL)),
            parsed=parsed,
            raw=raw,
            answer=ans,
        )
    except (TypeError, ValueError) as e:
        return {
            "error": f"normalize_error:{e}",
            "provider": PROVIDER,
            "model": str(getattr(model, "name_or_path", DEFAULT_MODEL)),
            "raw_preview": raw[:400],
        }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dims = ("grounding", "relevance", "document_necessity", "overall")
    teacher_tiers: list[str] = []
    student_tiers: list[str] = []
    false_high = 0
    false_drop = 0
    parse_errors = 0
    dim_mae: dict[str, float] = {d: 0.0 for d in dims}
    dim_n = 0
    overall_teacher: list[float] = []
    overall_student: list[float] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        tj = row.get("teacher_judge") or {}
        sj = row.get("student_judge") or {}
        ans = str(row.get("answer") or "")
        t_tier = str(row.get("teacher_quality_tier") or teacher_quality_tier(tj, ans))
        if sj.get("error"):
            parse_errors += 1
            s_tier = "error"
        else:
            s_tier = quality_tier_from_scores(
                grounding=int(sj.get("grounding", 0)),
                relevance=int(sj.get("relevance", 0)),
                document_necessity=int(sj.get("document_necessity", 0)),
                overall=int(sj.get("overall", 0)),
                answer=ans,
            )
        teacher_tiers.append(t_tier)
        student_tiers.append(s_tier)
        confusion[t_tier][s_tier] += 1
        if t_tier != "high" and s_tier == "high":
            false_high += 1
        if t_tier == "drop" and s_tier != "drop":
            false_drop += 1

        if not sj.get("error"):
            ok = True
            for d in dims:
                if d not in tj or d not in sj:
                    ok = False
                    break
            if ok:
                dim_n += 1
                for d in dims:
                    dim_mae[d] += abs(int(sj[d]) - int(tj[d]))
                overall_teacher.append(float(tj["overall"]))
                overall_student.append(float(sj["overall"]))

    tier_agreement = (
        sum(1 for a, b in zip(teacher_tiers, student_tiers) if a == b) / len(rows) if rows else 0.0
    )
    mae = {d: (dim_mae[d] / dim_n if dim_n else None) for d in dims}
    return {
        "n_rows": len(rows),
        "n_scored_rows": dim_n,
        "n_parse_errors": parse_errors,
        "tier_agreement": tier_agreement,
        "cohen_kappa": cohen_kappa(teacher_tiers, student_tiers),
        "overall_spearman": spearman(overall_teacher, overall_student),
        "mean_abs_error": mae,
        "teacher_tier_counts": dict(Counter(teacher_tiers)),
        "student_tier_counts": dict(Counter(student_tiers)),
        "tier_confusion": {k: dict(v) for k, v in sorted(confusion.items())},
        "false_high_rate": false_high / len(rows) if rows else 0.0,
        "false_drop_rate": false_drop / len(rows) if rows else 0.0,
    }


def run_eval_judge_filter(ns: argparse.Namespace) -> int:
    wall0 = time.perf_counter()
    test_path = Path(ns.test_jsonl).expanduser().resolve()
    adapter_dir = Path(ns.adapter_dir).expanduser().resolve() if ns.adapter_dir else None
    out_dir = Path(ns.out_dir).expanduser().resolve() if ns.out_dir else (adapter_dir.parent / "eval" if adapter_dir else Path.cwd() / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(test_path)
    if int(ns.max_rows) > 0:
        rows = rows[: int(ns.max_rows)]
    if not rows:
        print(f"No rows in {test_path}", file=sys.stderr)
        return 1

    use_bf16 = not ns.no_bf16
    print(f"Loading model base={ns.model} adapter={adapter_dir}", flush=True)
    model, tokenizer = load_judge_model(
        base_model=str(ns.model),
        adapter_dir=adapter_dir,
        use_bf16=use_bf16,
    )

    scored: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, row in enumerate(rows, 1):
        sj = predict_judge_block(
            model,
            tokenizer,
            row,
            max_context_chars=int(ns.max_context_chars),
            max_new_tokens=int(ns.max_new_tokens),
            use_bf16=use_bf16,
        )
        out_row = dict(row)
        out_row["student_judge"] = sj
        scored.append(out_row)
        if i % 50 == 0 or i == len(rows):
            print(f"  ... {i}/{len(rows)} scored", flush=True)

    infer_wall_s = time.perf_counter() - t0
    stats = summarize(scored)
    summary = {
        "schema": "judge_filter_eval_summary/v1",
        "test_jsonl": str(test_path),
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "base_model": str(ns.model),
        "max_context_chars": int(ns.max_context_chars),
        "max_new_tokens": int(ns.max_new_tokens),
        "infer_wall_s": infer_wall_s,
        "total_wall_s": time.perf_counter() - wall0,
        "stats": stats,
    }

    pred_path = out_dir / "test_predictions.jsonl"
    summary_path = out_dir / "eval_summary.json"
    write_jsonl(pred_path, scored)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {pred_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    return 0
