"""
Document-level coverage of synthetic QAs over parent documents.

Splits each parent document into N equal windows; a window is covered if the
aggregated Q+A text from that document overlaps the window above a threshold.

Modes:
  - lexical: token-set overlap ratio
  - embedding: MiniLM cosine similarity (vector space; batched encode)

  python -m thesis.cli analyze-document-coverage \\
    --docs-jsonl .../documents_unique.jsonl \\
    --qa-jsonl .../synthetic_qa_haiku_judge.jsonl \\
    --out-dir .../analysis/coverage_embed \\
    --mode embedding --exclude-tiers drop --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LEXICAL_THR = 0.12
DEFAULT_EMBED_THR = 0.35


def _tok(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _words(s: str) -> list[str]:
    return re.findall(r"\S+", (s or "").strip())


def _qa_tier(row: dict[str, Any]) -> str:
    return str(
        (row.get("llm_judge") or {}).get("quality_tier") or row.get("quality_tier") or "unknown"
    )


def _parent_doc_id(row: dict[str, Any]) -> str:
    did = (row.get("document_id") or "").strip()
    if did:
        return did
    sid = (row.get("section_id") or row.get("chunk_id") or "").strip()
    if "::" in sid:
        return sid.split("::", 1)[0]
    return sid or "unknown"


def _split_windows(context: str, n_windows: int) -> list[str]:
    words = _words(context)
    if len(words) < 40:
        return []
    wsize = max(len(words) // n_windows, 1)
    windows: list[str] = []
    for i in range(n_windows):
        start = i * wsize
        end = len(words) if i == n_windows - 1 else (i + 1) * wsize
        windows.append(" ".join(words[start:end]))
    return windows


def _window_hits_lexical(
    context: str,
    qa_text: str,
    *,
    n_windows: int,
    thr: float,
) -> list[bool]:
    windows = _split_windows(context, n_windows)
    if not windows:
        return []
    qa = _tok(qa_text)
    hits: list[bool] = []
    for wtxt in windows:
        wt = _tok(wtxt)
        if not wt or len(qa) < 3:
            hits.append(False)
            continue
        hits.append(len(qa & wt) / len(wt) >= thr)
    return hits


def _load_embed_model(embed_model_name: str, device: str) -> Any:
    from sentence_transformers import SentenceTransformer

    load_kwargs: dict[str, Any] = {"device": device}
    local_snap = Path.home() / (
        ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
    )
    model_path = embed_model_name
    if embed_model_name.endswith("all-MiniLM-L6-v2") and local_snap.is_dir():
        snaps = sorted(p for p in local_snap.iterdir() if p.is_dir())
        if snaps:
            model_path = str(snaps[0])
            load_kwargs["local_files_only"] = True
            print(f"  using local cache: {model_path}", flush=True)
    try:
        return SentenceTransformer(model_path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("local_files_only", None)
        return SentenceTransformer(model_path, **load_kwargs)


def _score_docs_embedding(
    scored_docs: list[dict[str, Any]],
    *,
    model: Any,
    thr: float,
    batch_size: int,
) -> None:
    """In-place: add hits/sims to each scored_docs entry via one batched encode."""
    import numpy as np

    win_texts: list[str] = []
    qa_texts: list[str] = []
    for item in scored_docs:
        win_texts.extend(item["windows"])
        qa_texts.append(item["qa_blob"])

    print(
        f"Encoding {len(win_texts)} windows + {len(qa_texts)} QA blobs "
        f"(batch_size={batch_size}) …",
        flush=True,
    )
    win_emb = model.encode(
        win_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    qa_emb = model.encode(
        qa_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    offset = 0
    for i, item in enumerate(scored_docs):
        n = len(item["windows"])
        w = win_emb[offset : offset + n]
        offset += n
        sims = (w @ qa_emb[i]).astype(float)
        sims_list = [float(x) for x in np.asarray(sims).tolist()]
        item["sims"] = sims_list
        item["hits"] = [s >= thr for s in sims_list]


def run_analyze_document_coverage(ns: argparse.Namespace) -> int:
    docs_path = Path(ns.docs_jsonl).expanduser().resolve()
    qa_path = Path(ns.qa_jsonl).expanduser().resolve()
    out_dir = Path(ns.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = str(getattr(ns, "mode", "lexical") or "lexical").strip().lower()
    if mode not in ("lexical", "embedding"):
        raise SystemExit("--mode must be lexical or embedding")

    docs: dict[str, dict[str, Any]] = {}
    for line in docs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        docs[str(r["document_id"])] = r

    exclude_tiers = {
        str(t).strip().lower() for t in (getattr(ns, "exclude_tiers", None) or []) if str(t).strip()
    }
    by_doc_qa: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tier_counts: dict[str, int] = defaultdict(int)
    n_excluded = 0
    for line in qa_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tier = _qa_tier(r)
        if tier.lower() in exclude_tiers:
            n_excluded += 1
            continue
        did = _parent_doc_id(r)
        by_doc_qa[did].append(r)
        tier_counts[tier] += 1

    n_windows = int(ns.n_windows)
    if mode == "embedding":
        thr = float(getattr(ns, "cosine_threshold", None) or DEFAULT_EMBED_THR)
    else:
        thr = float(ns.overlap_threshold)

    embed_model_name = str(getattr(ns, "embed_model", None) or DEFAULT_EMBED_MODEL)
    batch_size = int(getattr(ns, "batch_size", 128) or 128)
    device = str(getattr(ns, "device", "auto") or "auto")

    # Build per-doc work list once
    work: list[dict[str, Any]] = []
    for did, doc in docs.items():
        rows = by_doc_qa.get(did, [])
        windows = _split_windows(doc.get("context") or "", n_windows)
        if not windows:
            continue
        qa_blob = " ".join(f"{r.get('question') or ''} {r.get('answer') or ''}" for r in rows)
        tiers = [_qa_tier(r) for r in rows]
        maj = max(set(tiers), key=tiers.count) if tiers else "none"
        work.append(
            {
                "document_id": did,
                "n_qa": len(rows),
                "windows": windows,
                "qa_blob": qa_blob,
                "majority_tier": maj,
            }
        )

    if mode == "embedding":
        try:
            import torch
        except ImportError as e:
            raise SystemExit("Missing dependency: torch") from e
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "Missing dependency: sentence-transformers\n"
                "  pip install sentence-transformers"
            ) from e
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embed model {embed_model_name} on {device} …", flush=True)
        model = _load_embed_model(embed_model_name, device)
        _score_docs_embedding(work, model=model, thr=thr, batch_size=batch_size)
    else:
        for item in work:
            qa = _tok(item["qa_blob"])
            hits = []
            for wtxt in item["windows"]:
                wt = _tok(wtxt)
                if not wt or len(qa) < 3:
                    hits.append(False)
                else:
                    hits.append(len(qa & wt) / len(wt) >= thr)
            item["hits"] = hits
            item["sims"] = None

    per_doc: list[dict[str, Any]] = []
    coverages: list[float] = []
    all_sims: list[float] = []
    window_hits_global = [0] * n_windows
    window_tot_global = [0] * n_windows
    by_tier_cov: dict[str, list[float]] = defaultdict(list)

    for item in work:
        hits = item["hits"]
        cov = sum(1 for h in hits if h) / len(hits)
        coverages.append(cov)
        maj = item["majority_tier"]
        by_tier_cov[maj].append(cov)
        for i, h in enumerate(hits):
            window_tot_global[i] += 1
            if h:
                window_hits_global[i] += 1
        row_out: dict[str, Any] = {
            "document_id": item["document_id"],
            "n_qa": item["n_qa"],
            "coverage": round(cov, 4),
            "window_hits": hits,
            "majority_tier": maj,
        }
        sims = item.get("sims")
        if sims is not None:
            all_sims.extend(sims)
            row_out["window_cosines"] = [round(s, 4) for s in sims]
            row_out["mean_window_cosine"] = round(statistics.mean(sims), 4)
        per_doc.append(row_out)

    def pct(xs: list[float], t: float) -> float:
        if not xs:
            return 0.0
        return 100.0 * sum(1 for x in xs if x >= t) / len(xs)

    summary: dict[str, Any] = {
        "schema": "document_coverage_summary/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "docs_jsonl": str(docs_path),
        "qa_jsonl": str(qa_path),
        "settings": {
            "n_windows": n_windows,
            "overlap_threshold": thr,
            "mode": "embedding_cosine" if mode == "embedding" else "lexical_window",
            "exclude_tiers": sorted(exclude_tiers),
        },
        "n_qa_rows_excluded": n_excluded,
        "n_documents_scored": len(coverages),
        "n_qa_rows": sum(len(v) for v in by_doc_qa.values()),
        "tier_counts_qa": dict(tier_counts),
        "mean_coverage": round(statistics.mean(coverages), 4) if coverages else None,
        "median_coverage": round(statistics.median(coverages), 4) if coverages else None,
        "pct_docs_coverage_ge_80": round(pct(coverages, 0.8), 2),
        "pct_docs_coverage_ge_40": round(pct(coverages, 0.4), 2),
        "coverage_by_window_position": {
            f"window_{i+1}": round(
                100.0 * window_hits_global[i] / max(window_tot_global[i], 1), 2
            )
            for i in range(n_windows)
        },
        "mean_coverage_by_majority_tier": {
            k: round(statistics.mean(v), 4) for k, v in sorted(by_tier_cov.items()) if v
        },
    }
    if mode == "embedding":
        summary["settings"]["embed_model"] = embed_model_name
        summary["settings"]["cosine_threshold"] = thr
        summary["settings"]["device"] = device
        summary["settings"]["batch_size"] = batch_size
        if all_sims:
            summary["mean_window_cosine"] = round(statistics.mean(all_sims), 4)
            summary["median_window_cosine"] = round(statistics.median(all_sims), 4)

    (out_dir / "document_coverage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (out_dir / "document_coverage_per_doc.jsonl").open("w", encoding="utf-8") as fp:
        for row in per_doc:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'document_coverage_summary.json'}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Document coverage of synthetic QAs")
    p.add_argument("--docs-jsonl", type=Path, required=True)
    p.add_argument("--qa-jsonl", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-windows", type=int, default=5)
    p.add_argument("--overlap-threshold", type=float, default=DEFAULT_LEXICAL_THR)
    p.add_argument(
        "--mode",
        choices=("lexical", "embedding"),
        default="lexical",
        help="lexical token overlap, or embedding cosine (MiniLM).",
    )
    p.add_argument(
        "--cosine-threshold",
        type=float,
        default=DEFAULT_EMBED_THR,
        help="Window covered if QA↔window cosine >= this (embedding mode).",
    )
    p.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--exclude-tiers",
        nargs="+",
        default=None,
        help="Skip QA rows with these judge tiers (e.g. drop).",
    )
    return p


if __name__ == "__main__":
    raise SystemExit(run_analyze_document_coverage(build_arg_parser().parse_args()))
