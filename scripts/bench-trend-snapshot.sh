#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bench-trend-snapshot.sh [options]

Generate trend snapshots from matrix gate artifacts.

Options:
  --matrix-root PATH   Matrix root directory (default: ops/benchmarks/matrix)
  --out-csv PATH       Output CSV path (default: ops/benchmarks/trends/matrix-gates.csv)
  --out-json PATH      Output JSON summary path (default: ops/benchmarks/trends/matrix-gates.json)
  --limit N            Keep latest N matrix runs in outputs (default: 50)
  --help               Show this help
USAGE
}

MATRIX_ROOT="ops/benchmarks/matrix"
OUT_CSV="ops/benchmarks/trends/matrix-gates.csv"
OUT_JSON="ops/benchmarks/trends/matrix-gates.json"
LIMIT=50

while [[ $# -gt 0 ]]; do
  case "$1" in
    --matrix-root) MATRIX_ROOT="$2"; shift 2 ;;
    --out-csv) OUT_CSV="$2"; shift 2 ;;
    --out-json) OUT_JSON="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]] || (( LIMIT <= 0 )); then
  echo "Invalid --limit: ${LIMIT}" >&2
  exit 2
fi

for cmd in jq find sort head mkdir dirname mktemp awk; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 127
  fi
done

mkdir -p "$(dirname "${OUT_CSV}")"
mkdir -p "$(dirname "${OUT_JSON}")"

tmp_rows="$(mktemp)"
tmp_jsonl="$(mktemp)"

find "${MATRIX_ROOT}" -type f -name gate-summary.json 2>/dev/null | while read -r file; do
  matrix_name="$(jq -r '.matrix_name // "unknown"' "${file}")"
  generated_at="$(jq -r '.generated_at_utc // "unknown"' "${file}")"
  matrix_pass="$(jq -r '.matrix_pass // false' "${file}")"
  req_delta="$(jq -r '.observed.hybrid_req_delta_pct // 0' "${file}")"
  tok_delta="$(jq -r '.observed.hybrid_tokens_delta_pct // 0' "${file}")"
  lat_delta="$(jq -r '.observed.hybrid_latency_p95_delta_pct // 0' "${file}")"
  err_delta="$(jq -r '.observed.hybrid_error_delta_abs // 0' "${file}")"
  g_thr="$(jq -r '.gates.hybrid_throughput // false' "${file}")"
  g_lat="$(jq -r '.gates.hybrid_latency // false' "${file}")"
  g_err="$(jq -r '.gates.hybrid_error // false' "${file}")"
  g_sa2_rt="$(jq -r '.gates.sa2_runtime_required_pass // false' "${file}")"
  rt_gate_mode="$(jq -r '.thresholds.runtime_tag_gate_mode // "unknown"' "${file}")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${generated_at}" "${matrix_name}" "${matrix_pass}" "${req_delta}" "${tok_delta}" "${lat_delta}" "${err_delta}" \
    "${g_thr}" "${g_lat}" "${g_err}" "${g_sa2_rt}" "${rt_gate_mode}" >> "${tmp_rows}"

  jq -nc \
    --arg generated_at_utc "${generated_at}" \
    --arg matrix_name "${matrix_name}" \
    --arg matrix_pass "${matrix_pass}" \
    --arg req_delta "${req_delta}" \
    --arg tok_delta "${tok_delta}" \
    --arg lat_delta "${lat_delta}" \
    --arg err_delta "${err_delta}" \
    --arg g_thr "${g_thr}" \
    --arg g_lat "${g_lat}" \
    --arg g_err "${g_err}" \
    --arg g_sa2_rt "${g_sa2_rt}" \
    --arg rt_gate_mode "${rt_gate_mode}" \
    '{
      generated_at_utc: $generated_at_utc,
      matrix_name: $matrix_name,
      matrix_pass: ($matrix_pass == "true"),
      observed: {
        hybrid_req_delta_pct: ($req_delta | tonumber),
        hybrid_tokens_delta_pct: ($tok_delta | tonumber),
        hybrid_latency_p95_delta_pct: ($lat_delta | tonumber),
        hybrid_error_delta_abs: ($err_delta | tonumber)
      },
      gates: {
        hybrid_throughput: ($g_thr == "true"),
        hybrid_latency: ($g_lat == "true"),
        hybrid_error: ($g_err == "true"),
        sa2_runtime_required_pass: ($g_sa2_rt == "true")
      },
      runtime_tag_gate_mode: $rt_gate_mode
    }' >> "${tmp_jsonl}"
done

{
  echo "generated_at_utc,matrix_name,matrix_pass,hybrid_req_delta_pct,hybrid_tokens_delta_pct,hybrid_latency_p95_delta_pct,hybrid_error_delta_abs,gate_hybrid_throughput,gate_hybrid_latency,gate_hybrid_error,gate_sa2_runtime_required_pass,runtime_tag_gate_mode"
  if [[ -s "${tmp_rows}" ]]; then
    sort -r "${tmp_rows}" | head -n "${LIMIT}" | awk -F'\t' '{
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
    }'
  fi
} > "${OUT_CSV}"

if [[ -s "${tmp_jsonl}" ]]; then
  jq -s --arg generated_at "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" --arg matrix_root "${MATRIX_ROOT}" --argjson limit "${LIMIT}" '
    sort_by(.generated_at_utc) | reverse | .[:$limit] as $recent
    | {
        generated_at_utc: $generated_at,
        matrix_root: $matrix_root,
        limit: $limit,
        total_detected: length,
        recent: $recent,
        summary: {
          pass_count: ($recent | map(select(.matrix_pass == true)) | length),
          fail_count: ($recent | map(select(.matrix_pass == false)) | length),
          avg_hybrid_req_delta_pct: (($recent | map(.observed.hybrid_req_delta_pct) | add) / (($recent | length) | if . == 0 then 1 else . end)),
          avg_hybrid_tokens_delta_pct: (($recent | map(.observed.hybrid_tokens_delta_pct) | add) / (($recent | length) | if . == 0 then 1 else . end)),
          avg_hybrid_latency_p95_delta_pct: (($recent | map(.observed.hybrid_latency_p95_delta_pct) | add) / (($recent | length) | if . == 0 then 1 else . end))
        }
      }
  ' "${tmp_jsonl}" > "${OUT_JSON}"
else
  jq -n --arg generated_at "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" --arg matrix_root "${MATRIX_ROOT}" --argjson limit "${LIMIT}" '
    {
      generated_at_utc: $generated_at,
      matrix_root: $matrix_root,
      limit: $limit,
      total_detected: 0,
      recent: [],
      summary: {
        pass_count: 0,
        fail_count: 0,
        avg_hybrid_req_delta_pct: 0,
        avg_hybrid_tokens_delta_pct: 0,
        avg_hybrid_latency_p95_delta_pct: 0
      }
    }
  ' > "${OUT_JSON}"
fi

rm -f "${tmp_rows}" "${tmp_jsonl}"
echo "Trend snapshot generated:"
echo "  - ${OUT_CSV}"
echo "  - ${OUT_JSON}"
