#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bench-gate-alert.sh --gate-json PATH [--out-md PATH]

Parse matrix gate-summary.json and emit a concise alert summary.

Options:
  --gate-json PATH   Path to gate-summary.json (required)
  --out-md PATH      Optional markdown output file
  --help             Show this help
USAGE
}

GATE_JSON=""
OUT_MD=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gate-json) GATE_JSON="$2"; shift 2 ;;
    --out-md) OUT_MD="$2"; shift 2 ;;
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

matrix_name="$(jq -r '.matrix_name // "unknown"' "${GATE_JSON}")"
generated_at="$(jq -r '.generated_at_utc // "unknown"' "${GATE_JSON}")"
matrix_pass="$(jq -r '.matrix_pass // false' "${GATE_JSON}")"
req_delta="$(jq -r '.observed.hybrid_req_delta_pct // 0' "${GATE_JSON}")"
tok_delta="$(jq -r '.observed.hybrid_tokens_delta_pct // 0' "${GATE_JSON}")"
lat_delta="$(jq -r '.observed.hybrid_latency_p95_delta_pct // 0' "${GATE_JSON}")"
err_delta="$(jq -r '.observed.hybrid_error_delta_abs // 0' "${GATE_JSON}")"
g_thr="$(jq -r '.gates.hybrid_throughput // false' "${GATE_JSON}")"
g_lat="$(jq -r '.gates.hybrid_latency // false' "${GATE_JSON}")"
g_err="$(jq -r '.gates.hybrid_error // false' "${GATE_JSON}")"
g_sa2_rt="$(jq -r '.gates.sa2_runtime_required_pass // false' "${GATE_JSON}")"

if [[ "${matrix_pass}" == "true" ]]; then
  status="PASS"
else
  status="FAIL"
fi

summary_line="Matrix ${status}: ${matrix_name} | req_delta=${req_delta}% tok_delta=${tok_delta}% lat_p95_delta=${lat_delta}% err_delta=${err_delta}"
echo "${summary_line}"
echo "GATE_STATUS=${status}"
echo "MATRIX_NAME=${matrix_name}"

if [[ -n "${OUT_MD}" ]]; then
  {
    echo "## Matrix Gate Alert"
    echo ""
    echo "- status: \`${status}\`"
    echo "- matrix: \`${matrix_name}\`"
    echo "- generated_at_utc: \`${generated_at}\`"
    echo "- hybrid_req_delta_pct: \`${req_delta}\`"
    echo "- hybrid_tokens_delta_pct: \`${tok_delta}\`"
    echo "- hybrid_latency_p95_delta_pct: \`${lat_delta}\`"
    echo "- hybrid_error_delta_abs: \`${err_delta}\`"
    echo ""
    echo "### Gate Checks"
    echo "- hybrid_throughput: \`${g_thr}\`"
    echo "- hybrid_latency: \`${g_lat}\`"
    echo "- hybrid_error: \`${g_err}\`"
    echo "- sa2_runtime_required_pass: \`${g_sa2_rt}\`"
  } > "${OUT_MD}"
fi
