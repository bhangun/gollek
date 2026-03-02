#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bench-compare.sh --baseline PATH --candidate PATH [--candidate PATH ...] [options]

Compare Multi-LoRA benchmark summaries against a baseline run.

Options:
  --baseline PATH       Baseline run directory or summary.json file (required)
  --candidate PATH      Candidate run directory or summary.json file (repeatable, required)
  --out-dir PATH        Output directory (default: ops/benchmarks/comparisons)
  --name LABEL          Optional report label
  --help                Show this help

Examples:
  ./scripts/bench-compare.sh \
    --baseline ops/benchmarks/zipf-balanced-a1.0-n500-c32-20260228-010101 \
    --candidate ops/benchmarks/zipf-balanced-a1.0-n500-c32-20260228-020202
USAGE
}

BASELINE_PATH=""
CANDIDATES=()
OUT_DIR="ops/benchmarks/comparisons"
REPORT_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline) BASELINE_PATH="$2"; shift 2 ;;
    --candidate) CANDIDATES+=("$2"); shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --name) REPORT_NAME="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${BASELINE_PATH}" ]]; then
  echo "--baseline is required" >&2
  exit 2
fi
if (( ${#CANDIDATES[@]} == 0 )); then
  echo "At least one --candidate is required" >&2
  exit 2
fi

for cmd in jq awk date mktemp basename dirname mkdir; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 127
  fi
done

resolve_summary_file() {
  local path="$1"
  if [[ -d "${path}" ]]; then
    echo "${path%/}/summary.json"
  else
    echo "${path}"
  fi
}

is_number() {
  [[ "${1}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]
}

calc_delta_abs() {
  awk -v b="$1" -v c="$2" 'BEGIN { printf("%.6f", c - b) }'
}

calc_delta_pct() {
  awk -v b="$1" -v c="$2" 'BEGIN {
    if (b == 0) { printf("0.0000"); }
    else { printf("%.4f", ((c - b) / b) * 100.0); }
  }'
}

judge_delta() {
  local direction="$1"
  local delta="$2"
  if ! is_number "${delta}"; then
    echo "unknown"
    return
  fi
  awk -v d="${direction}" -v x="${delta}" 'BEGIN {
    eps = 0.0000001;
    if (x > -eps && x < eps) { print "same"; exit; }
    if (d == "higher") { print (x > 0 ? "better" : "worse"); }
    else { print (x < 0 ? "better" : "worse"); }
  }'
}

extract_string() {
  local file="$1"
  local expr="$2"
  jq -r "${expr} // \"unknown\"" "${file}"
}

extract_raw() {
  local file="$1"
  local expr="$2"
  jq -r "${expr} // empty" "${file}"
}

baseline_summary="$(resolve_summary_file "${BASELINE_PATH}")"
if [[ ! -f "${baseline_summary}" ]]; then
  echo "Baseline summary not found: ${baseline_summary}" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
if [[ -z "${REPORT_NAME}" ]]; then
  REPORT_NAME="compare-${timestamp}"
fi
report_dir="${OUT_DIR}/${REPORT_NAME}"
mkdir -p "${report_dir}"

report_txt="${report_dir}/report.txt"
report_json="${report_dir}/report.json"
rows_tsv="$(mktemp)"
jsonl_file="$(mktemp)"

baseline_run_id="$(extract_string "${baseline_summary}" '.run_id')"
baseline_profile="$(extract_string "${baseline_summary}" '.advanced_profile')"
baseline_mode="$(extract_string "${baseline_summary}" '.runtime_tags.advanced_attention_mode')"
baseline_sage="$(extract_string "${baseline_summary}" '.runtime_tags.advanced_sage_attention2_requested')"

metrics=(
  "throughput_req_s|higher"
  "throughput_tokens_s_est|higher"
  "latency_all_p95_ms|lower"
  "ttft_p95_ms|lower"
  "tpot_p95_ms|lower"
  "error_rate|lower"
  "adapter_switch_ratio|lower"
)

for candidate_path in "${CANDIDATES[@]}"; do
  candidate_summary="$(resolve_summary_file "${candidate_path}")"
  if [[ ! -f "${candidate_summary}" ]]; then
    echo "Candidate summary not found: ${candidate_summary}" >&2
    exit 2
  fi

  candidate_run_id="$(extract_string "${candidate_summary}" '.run_id')"
  candidate_profile="$(extract_string "${candidate_summary}" '.advanced_profile')"
  candidate_mode="$(extract_string "${candidate_summary}" '.runtime_tags.advanced_attention_mode')"
  candidate_sage="$(extract_string "${candidate_summary}" '.runtime_tags.advanced_sage_attention2_requested')"

  candidate_label="$(basename "$(dirname "${candidate_summary}")")"

  for metric_def in "${metrics[@]}"; do
    metric="${metric_def%%|*}"
    direction="${metric_def##*|}"
    base_val="$(extract_raw "${baseline_summary}" ".${metric}")"
    cand_val="$(extract_raw "${candidate_summary}" ".${metric}")"

    if is_number "${base_val}" && is_number "${cand_val}"; then
      delta_abs="$(calc_delta_abs "${base_val}" "${cand_val}")"
      delta_pct="$(calc_delta_pct "${base_val}" "${cand_val}")"
      verdict="$(judge_delta "${direction}" "${delta_abs}")"
    else
      delta_abs="unknown"
      delta_pct="unknown"
      verdict="unknown"
      base_val="${base_val:-unknown}"
      cand_val="${cand_val:-unknown}"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${candidate_label}" "${candidate_run_id}" "${candidate_profile}" "${candidate_mode}" "${candidate_sage}" \
      "${metric}" "${direction}" "${base_val}" "${cand_val}" "${delta_abs}" "${delta_pct}" "${verdict}" >> "${rows_tsv}"

    jq -nc \
      --arg candidate_label "${candidate_label}" \
      --arg candidate_run_id "${candidate_run_id}" \
      --arg candidate_profile "${candidate_profile}" \
      --arg candidate_mode "${candidate_mode}" \
      --arg candidate_sage "${candidate_sage}" \
      --arg metric "${metric}" \
      --arg direction "${direction}" \
      --arg baseline_value "${base_val}" \
      --arg candidate_value "${cand_val}" \
      --arg delta_abs "${delta_abs}" \
      --arg delta_pct "${delta_pct}" \
      --arg verdict "${verdict}" \
      '{
        candidate_label: $candidate_label,
        candidate_run_id: $candidate_run_id,
        candidate_profile: $candidate_profile,
        candidate_runtime_mode: $candidate_mode,
        candidate_sage_requested: $candidate_sage,
        metric: $metric,
        direction: $direction,
        baseline_value: $baseline_value,
        candidate_value: $candidate_value,
        delta_abs: $delta_abs,
        delta_pct: $delta_pct,
        verdict: $verdict
      }' >> "${jsonl_file}"
  done
done

{
  echo "report_name=${REPORT_NAME}"
  echo "generated_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "baseline_run_id=${baseline_run_id}"
  echo "baseline_profile=${baseline_profile}"
  echo "baseline_runtime_mode=${baseline_mode}"
  echo "baseline_sage_requested=${baseline_sage}"
  echo
  printf "%-36s %-28s %-24s %-8s %-26s %-8s %-12s %-12s %-12s %-12s %-8s\n" \
    "candidate" "candidate_run_id" "candidate_profile" "mode" "metric" "dir" "baseline" "candidate" "delta_abs" "delta_pct" "verdict"
  printf "%-36s %-28s %-24s %-8s %-26s %-8s %-12s %-12s %-12s %-12s %-8s\n" \
    "------------------------------------" "----------------------------" "------------------------" "--------" "--------------------------" "--------" "------------" "------------" "------------" "------------" "--------"
  awk -F'\t' '{
    printf "%-36s %-28s %-24s %-8s %-26s %-8s %-12s %-12s %-12s %-12s %-8s\n",
      $1, $2, $3, $4, $6, $7, $8, $9, $10, $11, $12
  }' "${rows_tsv}"
} > "${report_txt}"

jq -n \
  --arg report_name "${REPORT_NAME}" \
  --arg generated_at "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  --arg baseline_run_id "${baseline_run_id}" \
  --arg baseline_profile "${baseline_profile}" \
  --arg baseline_runtime_mode "${baseline_mode}" \
  --arg baseline_sage_requested "${baseline_sage}" \
  --slurpfile rows "${jsonl_file}" \
  '{
    report_name: $report_name,
    generated_at_utc: $generated_at,
    baseline: {
      run_id: $baseline_run_id,
      profile: $baseline_profile,
      runtime_mode: $baseline_runtime_mode,
      sage_requested: $baseline_sage_requested
    },
    comparisons: $rows
  }' > "${report_json}"

rm -f "${rows_tsv}" "${jsonl_file}"
echo "Comparison report generated:"
echo "  - ${report_txt}"
echo "  - ${report_json}"
