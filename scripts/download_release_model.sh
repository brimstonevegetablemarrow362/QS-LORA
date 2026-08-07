#!/usr/bin/env bash
# Unpack the GitHub Release LoRA zip (or use the local release/ folder).
#
# Usage:
#   scripts/download_release_model.sh /path/to/qa-generator-lora.zip
#   scripts/download_release_model.sh   # uses release/qa-generator-lora.zip if present

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZIP="${1:-${ROOT}/release/qa-generator-lora.zip}"
OUT="${ROOT}/out/qa-generator-lora"

if [[ ! -f "${ZIP}" ]]; then
  echo "Zip not found: ${ZIP}" >&2
  echo "Download the Release asset qa-generator-lora.zip, then re-run." >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT}")"
rm -rf "${OUT}"
unzip -q "${ZIP}" -d "$(dirname "${OUT}")"
# zip contains top-level qa-generator-lora/
if [[ ! -d "${OUT}" ]]; then
  echo "Expected ${OUT} after unzip" >&2
  exit 1
fi
echo "Adapter ready at: ${OUT}"
echo "Next: scripts/merge_generator.sh"
