#!/usr/bin/env bash
# Load AWS / Bedrock credentials from a local .env file (never commit real keys).
#
# Setup (once):
#   cp .env.example .env
#   chmod 600 .env
#   # edit .env and paste your AWS / Bedrock values
#
# Usage:
#   source scripts/source_bedrock_env.sh
#   python -m thesis.cli qa-bedrock-judge --predictions-jsonl ... --answer-field pred
#
# Override:
#   BEDROCK_ENV_FILE=/path/to/my.env source scripts/source_bedrock_env.sh

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
_DEFAULT_ENV_FILE="${_REPO_ROOT}/.env"
ENV_FILE="${BEDROCK_ENV_FILE:-${_DEFAULT_ENV_FILE}}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Credentials file not found: ${ENV_FILE}" >&2
  echo "" >&2
  echo "Create it from the example:" >&2
  echo "  cp ${_REPO_ROOT}/.env.example ${_REPO_ROOT}/.env" >&2
  echo "  chmod 600 ${_REPO_ROOT}/.env" >&2
  echo "  # set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, ..." >&2
  return 1 2>/dev/null || exit 1
fi

if [[ "$(uname)" != "MINGW"* && "$(uname)" != "MSYS"* ]]; then
  perm="$(stat -c '%a' "${ENV_FILE}" 2>/dev/null || stat -f '%OLp' "${ENV_FILE}" 2>/dev/null || echo '')"
  if [[ -n "${perm}" && "${perm}" != "600" && "${perm}" != "400" ]]; then
    echo "Warning: ${ENV_FILE} mode is ${perm}; recommend chmod 600" >&2
  fi
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

_iam_set=0
[[ -n "${AWS_ACCESS_KEY_ID:-}" && -n "${AWS_SECRET_ACCESS_KEY:-}" ]] && _iam_set=1

if [[ "${_iam_set}" -eq 1 && -n "${AWS_BEARER_TOKEN_BEDROCK:-}" ]]; then
  echo "Note: unsetting AWS_BEARER_TOKEN_BEDROCK (using IAM access key + secret)." >&2
  unset AWS_BEARER_TOKEN_BEDROCK
fi

_bearer_set=0
[[ -n "${AWS_BEARER_TOKEN_BEDROCK:-}" ]] && _bearer_set=1

_missing=()
[[ -z "${AWS_REGION:-}" && -z "${AWS_DEFAULT_REGION:-}" ]] && _missing+=("AWS_REGION")

if [[ "${_bearer_set}" -eq 0 ]]; then
  [[ -z "${AWS_ACCESS_KEY_ID:-}" ]] && _missing+=("AWS_ACCESS_KEY_ID")
  [[ -z "${AWS_SECRET_ACCESS_KEY:-}" ]] && _missing+=("AWS_SECRET_ACCESS_KEY")
fi

# Reject unedited placeholders from .env.example
if [[ "${AWS_ACCESS_KEY_ID:-}" == *REPLACE* || "${AWS_SECRET_ACCESS_KEY:-}" == *REPLACE* ]]; then
  echo "Edit ${ENV_FILE} and replace the dummy AWS_* placeholders." >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ${#_missing[@]} -gt 0 ]]; then
  echo "Missing required variables in ${ENV_FILE}:" >&2
  printf '  %s\n' "${_missing[@]}" >&2
  echo "  (Or set AWS_BEARER_TOKEN_BEDROCK for Bedrock API key auth.)" >&2
  return 1 2>/dev/null || exit 1
fi

export AWS_DEFAULT_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION}}"
export BEDROCK_ANTHROPIC_CEILING_MODEL_ID="${BEDROCK_ANTHROPIC_CEILING_MODEL_ID:-us.anthropic.claude-opus-4-8}"
export BEDROCK_NOVA_CEILING_MODEL_ID="${BEDROCK_NOVA_CEILING_MODEL_ID:-us.amazon.nova-2-lite-v1:0}"

echo "Loaded Bedrock env from: ${ENV_FILE}"
echo "  AWS_REGION=${AWS_REGION:-${AWS_DEFAULT_REGION}}"
echo "  BEDROCK_JUDGE_MODEL_ID=${BEDROCK_JUDGE_MODEL_ID:-<cli default>}"
if [[ "${_bearer_set}" -eq 1 ]]; then
  echo "  auth=AWS_BEARER_TOKEN_BEDROCK (set)"
elif [[ -n "${AWS_SESSION_TOKEN:-}" ]]; then
  echo "  auth=IAM (AWS_SESSION_TOKEN=set)"
else
  echo "  auth=IAM (access key + secret)"
fi
