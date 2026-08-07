# QS-LoRA + Document QA Generator

Public toolkit for:
1. **Turning uploaded documents into grounded Q/A pairs** (PDF → Markdown → chunks → local generator).
2. **Optional Bedrock judging** of those pairs (you supply AWS credentials via env).
3. **Quality-Stratified LoRA (QS-LoRA)** research code used in the thesis experiments.

## Quick start — generate QA from your docs

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# PDF uploads:
pip install -r requirements-docling.txt
# Merge / train (GPU):
pip install -r requirements-train.txt
# Then install vLLM matching your CUDA stack: https://docs.vllm.ai/
```

### 2. Get the QA generator (GitHub Release)

Download **`qa-generator-lora.zip`** from the [Releases](../../releases) page (or use the local copy under `release/` if you built this tree from source).

```bash
scripts/download_release_model.sh /path/to/qa-generator-lora.zip
# Accept Meta Llama license + set HF_TOKEN if needed, then:
scripts/merge_generator.sh
```

Base model: [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).  
The Release ships a **LoRA adapter only** (~90 MB zip) so we stay under GitHub’s 2 GB asset limit and avoid redistributing full Llama weights.

### 3. Serve the merged model

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./out/merged-qa-generator \
  --host 127.0.0.1 --port 8100 --dtype auto
```

### 4. Docs → Markdown → QA

```bash
# Markdown
scripts/docs_to_qa.sh examples/sample_docs/sample.md examples/qa/sample.jsonl

# PDF (Docling)
scripts/docs_to_qa.sh /path/to/paper.pdf examples/qa/paper.jsonl
```

Or step by step:

```bash
export PYTHONPATH=pipeline
python pipeline/convert_pdf_docling.py --input paper.pdf --out-dir ./md/
python pipeline/chunk_markdown.py --input ./md/ --out chunks.jsonl --source-basename
python pipeline/generate_qa_from_chunks.py \
  --chunks chunks.jsonl --out qa.jsonl \
  --vllm-base-url http://127.0.0.1:8100 \
  --vllm-model ./out/merged-qa-generator
```

### 5. Optional: judge QA pairs (AWS Bedrock)

```bash
cp .env.example .env
chmod 600 .env
# edit .env — replace dummy AWS_* values with your keys
source scripts/source_bedrock_env.sh

PYTHONPATH=. python -m thesis.cli qa-bedrock-judge \
  --predictions-jsonl examples/qa/sample.jsonl \
  --answer-field answer
```

No real credentials are in this repo. Judging is optional; generation is fully local once the adapter is merged.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `pipeline/` | PDF→MD, chunk markdown, generate QA, merge LoRA |
| `scripts/` | Helpers: env load, merge, docs→QA |
| `thesis/` | QS-LoRA train / eval / judge research package |
| `examples/` | Sample doc + compact result tables / coverage / general-ability |
| `docs/` | Method notes + full `RESULTS_SUMMARY.md` |
| `release/` | Build artifacts for GitHub Release (zip is gitignored) |

---

## Research results (highlights)

On RepLiQA / Quoref / SQuAD with Bedrock Haiku judges, **QS tier merge** beats uniform LoRA (B3) on gold-alignment. See:

- [`docs/results.md`](docs/results.md) — short tables
- [`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md) — full write-up
- [`examples/results/leaderboards/`](examples/results/leaderboards/) — JSON leaderboards
- [`examples/results/low_tier_slide_examples.jsonl`](examples/results/low_tier_slide_examples.jsonl) — low/drop synthetic QA examples

---

## License notes

- **Code** in this repository: MIT (see `LICENSE`).
- **QA generator adapter**: derivative of Llama 3.2; use under the [Meta Llama Community License](https://www.llama.com/llama3_2/license/). You must obtain base weights from Meta / Hugging Face yourself.
- **Datasets** (RepLiQA, SQuAD, Quoref, DROP): follow each dataset’s original license; we do not redistribute full train dumps here.

---

## Citing

If you use QS-LoRA or this generator pipeline, please cite the thesis / paper (add BibTeX when available) and link this repository.
