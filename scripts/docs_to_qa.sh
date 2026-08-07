#!/usr/bin/env bash
# End-to-end: PDF or Markdown → chunks → QA JSONL (needs vLLM generator running).
#
# Usage:
#   # Markdown only
#   scripts/docs_to_qa.sh examples/sample_docs/sample.md examples/qa/out.jsonl
#
#   # PDF (Docling)
#   scripts/docs_to_qa.sh /path/to/paper.pdf examples/qa/out.jsonl

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:?usage: docs_to_qa.sh <pdf|md|dir> [out.jsonl]}"
OUT="${2:-${ROOT}/examples/qa/all.jsonl}"
WORK="${ROOT}/examples/qa/_work"
mkdir -p "${WORK}" "$(dirname "${OUT}")"

export PYTHONPATH="${ROOT}/pipeline${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}/pipeline"

INPUT_PATH="$(cd "$(dirname "${INPUT}")" && pwd)/$(basename "${INPUT}")"
EXT="${INPUT_PATH##*.}"
EXT="$(echo "${EXT}" | tr '[:upper:]' '[:lower:]')"

MD_DIR="${WORK}/md"
CHUNKS="${WORK}/chunks.jsonl"
rm -rf "${MD_DIR}"
mkdir -p "${MD_DIR}"

if [[ -d "${INPUT_PATH}" ]]; then
  # Directory of PDFs and/or markdown
  shopt -s nullglob
  pdfs=("${INPUT_PATH}"/*.pdf "${INPUT_PATH}"/*.PDF)
  mds=("${INPUT_PATH}"/*.md "${INPUT_PATH}"/*.markdown)
  if ((${#pdfs[@]})); then
    python convert_pdf_docling.py --input "${INPUT_PATH}" --out-dir "${MD_DIR}"
  fi
  for f in "${mds[@]}"; do
    cp "${f}" "${MD_DIR}/"
  done
elif [[ "${EXT}" == "pdf" ]]; then
  python convert_pdf_docling.py --input "${INPUT_PATH}" --out-dir "${MD_DIR}"
elif [[ "${EXT}" == "md" || "${EXT}" == "markdown" ]]; then
  cp "${INPUT_PATH}" "${MD_DIR}/"
else
  echo "Unsupported input: ${INPUT_PATH} (use .pdf, .md, or a directory)" >&2
  exit 1
fi

python chunk_markdown.py --input "${MD_DIR}" --out "${CHUNKS}" --source-basename
python generate_qa_from_chunks.py \
  --chunks "${CHUNKS}" \
  --out "${OUT}" \
  --vllm-base-url "${GENERATOR_VLLM_URL:-http://${GENERATOR_VLLM_HOST:-127.0.0.1}:${GENERATOR_VLLM_PORT:-8100}}" \
  --vllm-model "${GENERATOR_VLLM_MODEL:-${ROOT}/out/merged-qa-generator}"

echo "Wrote QA pairs → ${OUT}"
