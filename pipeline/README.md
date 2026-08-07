# Document → QA pipeline

| Script | Step |
|--------|------|
| `convert_pdf_docling.py` | PDF → Markdown |
| `chunk_markdown.py` | Markdown → `chunks.jsonl` |
| `generate_qa_from_chunks.py` | chunks → QA JSONL (local **vLLM** generator) |
| `merge_lora.py` | LoRA adapter → merged folder for vLLM |
| `split_normalized_dataset.py` | optional train/val/test split of QA JSONL |
| `prompts.py` | generator / answerer prompt strings |

```bash
export PYTHONPATH=pipeline
python convert_pdf_docling.py --input doc.pdf --out-dir ./md/
python chunk_markdown.py --input ./md/ --out chunks.jsonl --source-basename
python generate_qa_from_chunks.py --chunks chunks.jsonl --out qa.jsonl \
  --vllm-base-url http://127.0.0.1:8100 \
  --vllm-model ../out/merged-qa-generator
```

Or use `scripts/docs_to_qa.sh` from the repo root.
