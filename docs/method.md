# Method (short)

## Document → QA pipeline

1. **PDF → Markdown** via Docling (`pipeline/convert_pdf_docling.py`).
2. **Markdown → chunks** on headings + packing (`pipeline/chunk_markdown.py`).
3. **Chunks → QA** with a local vLLM server serving the merged generator (`pipeline/generate_qa_from_chunks.py`).
4. **Optional judge** with Claude Haiku on AWS Bedrock (`thesis` Bedrock judge tools).

The generator is a LoRA SFT of `Llama-3.2-3B-Instruct` trained to emit grounded `{question, answer}` JSON from an excerpt.

## QS-LoRA (research)

1. Generate / collect synthetic QA; **tier** with an LLM judge (high / medium / low / drop).
2. Train **separate LoRAs** per kept tier (often different ranks, e.g. r=32/16/8).
3. **Merge** adapters with fixed weights (e.g. 0.6 / 0.3 / 0.1) into one dense checkpoint (`thesis/merge_qs_lora.py`).
4. Evaluate with the same Bedrock rubric used for baselines (uniform LoRA, AdaLoRA, etc.).

See `docs/RESULTS_SUMMARY.md` for ablations, listwise ranks, and cost notes.
