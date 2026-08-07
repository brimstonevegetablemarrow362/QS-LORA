#!/usr/bin/env python3
"""
PDF → Markdown using Docling (same core stack as ../chunking.py, without VLM/Nougat).

Use this when uploads are PDFs; then run chunk_markdown.py on the produced .md files.

Install (GPU optional for Docling; OCR may use CPU):
  pip install -r requirements-docling.txt

Usage:
  python convert_pdf_docling.py --input report.pdf --out report.md
  python convert_pdf_docling.py --input ./pdfs/ --out-dir ./uploads/
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def clean_markdown(md: str) -> str:
    md = md.replace("\r", "")
    md = md.replace("\n\n\n", "\n\n")
    return md


def build_converter():
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        TableFormerMode,
        TableStructureOptions,
        ThreadedPdfPipelineOptions,
    )

    # Fast defaults for research / published PDFs:
    # - OCR off by default (most papers have a text layer)
    # - Optional GPU acceleration via env var
    do_ocr = os.environ.get("DOCLING_DO_OCR", "0").strip().lower() in ("1", "true", "yes", "on")
    images_scale = float(os.environ.get("DOCLING_IMAGES_SCALE", "2.0"))
    layout_batch_size = int(os.environ.get("DOCLING_LAYOUT_BATCH_SIZE", "8"))
    num_threads = int(os.environ.get("DOCLING_NUM_THREADS", "9"))
    accel = os.environ.get("DOCLING_ACCEL", "auto").strip().lower()

    if accel in ("cuda", "gpu"):
        # Prefer CUDA, but fall back safely if not available.
        try:
            import torch

            device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
        except Exception:
            device = AcceleratorDevice.CPU
    elif accel in ("cpu",):
        device = AcceleratorDevice.CPU
    else:
        # "auto": prefer CUDA if available, otherwise CPU.
        try:
            import torch

            device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
        except Exception:
            device = AcceleratorDevice.CPU

    accel_options = AcceleratorOptions(device=device, num_threads=num_threads)
    table_options = TableStructureOptions(mode=TableFormerMode.FAST)

    pipeline_options = ThreadedPdfPipelineOptions(
        accelerator_options=accel_options,
        do_ocr=do_ocr,
        generate_picture_images=True,
        images_scale=images_scale,
        table_structure_options=table_options,
        layout_batch_size=layout_batch_size,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )


def pdf_to_markdown(pdf_path: Path, converter) -> str:
    from docling_core.types.doc import ImageRefMode

    result = converter.convert(str(pdf_path))
    doc = result.document
    md = doc.export_to_markdown(
        image_mode=ImageRefMode.PLACEHOLDER,
        include_annotations=True,
    )
    return clean_markdown(md)


def collect_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF: {path}")
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    raise FileNotFoundError(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Docling PDF → Markdown")
    ap.add_argument("--input", type=Path, required=True, help="PDF file or directory of PDFs")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", type=Path, help="Output .md path (single PDF input only)")
    g.add_argument("--out-dir", type=Path, dest="out_dir", help="Directory for .md files (one per PDF)")
    args = ap.parse_args()

    pdfs = collect_pdfs(args.input)
    if not pdfs:
        raise SystemExit(f"No PDFs found under {args.input}")

    if args.out is not None:
        if len(pdfs) != 1:
            raise SystemExit("--out requires exactly one PDF as --input")
        out_md = args.out
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Loading Docling converter... "
        f"(DOCLING_ACCEL={os.environ.get('DOCLING_ACCEL','auto')}, "
        f"DOCLING_DO_OCR={os.environ.get('DOCLING_DO_OCR','0')})",
        flush=True,
    )
    converter = build_converter()

    for pdf in pdfs:
        if args.out is not None:
            out_md = args.out
        else:
            assert args.out_dir is not None
            out_md = args.out_dir / (pdf.stem + ".md")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        print(f"Converting {pdf.name} ...", flush=True)
        md = pdf_to_markdown(pdf, converter)
        out_md.write_text(md, encoding="utf-8")
        print(f"  -> {out_md}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
