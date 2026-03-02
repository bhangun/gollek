#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: publish-latest-gate-summary.sh --gate-json PATH [--output PATH]

Convert a matrix gate-summary.json into website-friendly latest-gate-summary.json.

Options:
  --gate-json PATH   Path to gate-summary.json (required)
  --output PATH      Output JSON path
                     (default: ops/benchmarks/latest-gate-summary.json)
  --help             Show this help
USAGE
}

GATE_JSON=""
OUTPUT_PATH="ops/benchmarks/latest-gate-summary.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gate-json) GATE_JSON="$2"; shift 2 ;;
    --output) OUTPUT_PATH="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${GATE_JSON}" ]]; then
  echo "--gate-json is required" >&2
  exit 2
fi
if [[ ! -f "${GATE_JSON}" ]]; then
  echo "gate json not found: ${GATE_JSON}" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "Missing required command: jq" >&2
  exit 127
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"

jq -n \
  --arg generated_at "$(jq -r '.generated_at_utc // "unknown"' "${GATE_JSON}")" \
  --arg matrix_name "$(jq -r '.matrix_name // "unknown"' "${GATE_JSON}")" \
  --arg gate_status "$(jq -r 'if .matrix_pass == true then "PASS" else "FAIL" end' "${GATE_JSON}")" \
  --arg req "$(jq -r '.observed.hybrid_req_delta_pct // "n/a"' "${GATE_JSON}")" \
  --arg tok "$(jq -r '.observed.hybrid_tokens_delta_pct // "n/a"' "${GATE_JSON}")" \
  --arg lat "$(jq -r '.observed.hybrid_latency_p95_delta_pct // "n/a"' "${GATE_JSON}")" \
  --arg err "$(jq -r '.observed.hybrid_error_delta_abs // "n/a"' "${GATE_JSON}")" \
  --arg source "${GATE_JSON}" \
  '{
    gate_status: $gate_status,
    matrix_name: $matrix_name,
    generated_at_utc: $generated_at,
    hybrid_req_delta_pct: $req,
    hybrid_tokens_delta_pct: $tok,
    hybrid_latency_p95_delta_pct: $lat,
    hybrid_error_delta_abs: $err,
    source: $source
  }' > "${OUTPUT_PATH}"

echo "Published latest gate summary:"
echo "  - ${OUTPUT_PATH}"
