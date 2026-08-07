"""vLLM OpenAI-compatible client helpers for eval generation."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from thesis.eval_repliqa_generate import (
    SYSTEM_PROMPT_CTX,
    SYSTEM_PROMPT_NO_CTX,
    build_user_block,
)


def openai_base_url(base_url: str) -> str:
    bu = base_url.rstrip("/")
    if not bu.endswith("/v1"):
        bu = bu + "/v1"
    return bu


def check_vllm(client: Any) -> None:
    try:
        client.models.list()
    except Exception as e:
        raise SystemExit(
            f"Cannot reach vLLM OpenAI API: {e}\n"
            "Start the server first, e.g.:\n"
            "  python -m vllm.entrypoints.openai.api_server --model <path> --host 127.0.0.1 --port 8100"
        ) from e


def eval_messages_for_row(
    row: dict[str, Any],
    *,
    use_context: bool,
    context_fraction: float = 1.0,
) -> list[dict[str, str]]:
    system = SYSTEM_PROMPT_CTX if use_context else SYSTEM_PROMPT_NO_CTX
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": build_user_block(
                row, use_context=use_context, context_fraction=context_fraction
            ),
        },
    ]


def vllm_chat(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 0.95,
    seed: int | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": float(temperature),
    }
    if float(temperature) > 0.0:
        kwargs["top_p"] = float(top_p)
    if seed is not None:
        kwargs["seed"] = int(seed)
    r = client.chat.completions.create(**kwargs)
    return (r.choices[0].message.content or "").strip()


def generate_rows_vllm(
    *,
    client: Any,
    model: str,
    rows: list[dict[str, Any]],
    use_context: bool,
    context_fraction: float,
    max_new_tokens: int,
    concurrency: int = 4,
    temperature: float = 0.0,
    top_p: float = 0.95,
    seed: int | None = None,
    row_start: int = 0,
) -> tuple[list[str], list[float]]:
    if not rows:
        return [], []

    concurrency = max(1, int(concurrency))
    preds: list[str | None] = [None] * len(rows)
    times: list[float | None] = [None] * len(rows)

    def one(i: int) -> tuple[int, str, float]:
        t0 = time.perf_counter()
        pred = vllm_chat(
            client=client,
            model=model,
            messages=eval_messages_for_row(
                rows[i],
                use_context=use_context,
                context_fraction=context_fraction,
            ),
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=(int(seed) + row_start + i) if seed is not None else None,
        )
        return i, pred, time.perf_counter() - t0

    if concurrency == 1:
        for i in range(len(rows)):
            idx, pred, dt = one(i)
            preds[idx] = pred
            times[idx] = dt
            if (i + 1) % 20 == 0 or i + 1 == len(rows):
                print(f"  generated {i + 1}/{len(rows)}", flush=True)
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(one, i) for i in range(len(rows))]
            for fut in as_completed(futs):
                idx, pred, dt = fut.result()
                preds[idx] = pred
                times[idx] = dt
                done += 1
                if done % 20 == 0 or done == len(rows):
                    print(f"  generated {done}/{len(rows)}", flush=True)

    return [p or "" for p in preds], [float(t or 0.0) for t in times]
