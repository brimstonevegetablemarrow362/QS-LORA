#!/usr/bin/env bash
# Merge the QA-generator LoRA into Llama-3.2-3B-Instruct for vLLM.
#
# Requires: HF access to meta-llama/Llama-3.2-3B-Instruct (accept license + HF_TOKEN).
# GPU recommended.
#
# Usage:
#   source .env   # optional HF_TOKEN
#   scripts/merge_generator.sh
#   scripts/merge_generator.sh /path/to/adapter /path/to/out

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADAPTER="${1:-${ROOT}/out/qa-generator-lora}"
OUT="${2:-${ROOT}/out/merged-qa-generator}"
BASE="${GENERATOR_BASE_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"

if [[ ! -f "${ADAPTER}/adapter_model.safetensors" ]]; then
  echo "Adapter not found at ${ADAPTER}" >&2
  echo "Run scripts/download_release_model.sh first." >&2
  exit 1
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python pipeline/merge_lora.py \
  --base "${BASE}" \
  --adapter "${ADAPTER}" \
  --out "${OUT}" \
  --bf16

echo ""
echo "Merged weights: ${OUT}"
echo "Serve with vLLM, e.g.:"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "    --model ${OUT} --host 127.0.0.1 --port 8100 --dtype auto"
