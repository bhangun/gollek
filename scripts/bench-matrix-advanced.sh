#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bench-matrix-advanced.sh [options]

Run baseline/hybrid/SA2-intent benchmark matrix and evaluate promotion gates.

Options:
  --endpoint URL                    Inference endpoint (default: http://localhost:8080/api/v1/inference)
  --model ID                        Model id (default: llama-3-8b)
  --requests N                      Requests per profile (default: 200)
  --concurrency N                   Concurrency per profile (default: 16)
  --adapters N                      Adapter cardinality (default: 128)
  --zipf-alpha F                    Zipf alpha (default: 1.0)
  --mix NAME                        prefill-heavy|balanced|decode-heavy (default: balanced)
  --seed N                          Seed for reproducibility (default: 42)
  --api-key KEY                     Optional API key
  --api-key-header NAME             API key header (default: X-API-Key)
  --adapter-param-key NAME          Adapter key in payload (default: adapter_id)
  --health-url URL                  Optional runtime health URL
  --runtime-tag-gate MODE           Runtime-tag gate mode: auto|on|off (default: auto)
  --telemetry MODE                  auto|on|off (default: auto)
  --gpu-telemetry MODE              auto|on|off (default: auto)
  --sample-interval-sec N           Telemetry sample interval seconds (default: 1)
  --out-dir PATH                    Matrix root output (default: ops/benchmarks/matrix)
  --name LABEL                      Matrix run label (default: matrix-<timestamp>)
  --bench-script PATH               Benchmark script path (default: scripts/bench-multilora-zipf.sh)
  --compare-script PATH             Compare script path (default: scripts/bench-compare.sh)
  --hybrid-min-throughput-pct F     Required min uplift (%) for hybrid (default: 20)
  --hybrid-max-latency-regress-pct F Max allowed p95 latency regression (%) for hybrid (default: 10)
  --hybrid-max-error-regress-abs F  Max allowed absolute error-rate increase (default: 0.005)
  --help                            Show this help
USAGE
}

ENDPOINT="http://localhost:8080/api/v1/inference"
MODEL_ID="llama-3-8b"
REQUESTS=200
CONCURRENCY=16
ADAPTERS=128
ZIPF_ALPHA="1.0"
MIX="balanced"
SEED=42
API_KEY=""
API_KEY_HEADER="X-API-Key"
ADAPTER_PARAM_KEY="adapter_id"
HEALTH_URL=""
RUNTIME_TAG_GATE_MODE="auto"
TELEMETRY_MODE="auto"
GPU_TELEMETRY_MODE="auto"
SAMPLE_INTERVAL_SEC=1
OUT_DIR="ops/benchmarks/matrix"
RUN_NAME=""
BENCH_SCRIPT="scripts/bench-multilora-zipf.sh"
COMPARE_SCRIPT="scripts/bench-compare.sh"
HYBRID_MIN_THROUGHPUT_PCT="20"
HYBRID_MAX_LATENCY_REGRESS_PCT="10"
HYBRID_MAX_ERROR_REGRESS_ABS="0.005"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --model) MODEL_ID="$2"; shift 2 ;;
    --requests) REQUESTS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --adapters) ADAPTERS="$2"; shift 2 ;;
    --zipf-alpha) ZIPF_ALPHA="$2"; shift 2 ;;
    --mix) MIX="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --api-key-header) API_KEY_HEADER="$2"; shift 2 ;;
    --adapter-param-key) ADAPTER_PARAM_KEY="$2"; shift 2 ;;
    --health-url) HEALTH_URL="$2"; shift 2 ;;
    --runtime-tag-gate) RUNTIME_TAG_GATE_MODE="$2"; shift 2 ;;
    --telemetry) TELEMETRY_MODE="$2"; shift 2 ;;
    --gpu-telemetry) GPU_TELEMETRY_MODE="$2"; shift 2 ;;
    --sample-interval-sec) SAMPLE_INTERVAL_SEC="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --name) RUN_NAME="$2"; shift 2 ;;
    --bench-script) BENCH_SCRIPT="$2"; shift 2 ;;
    --compare-script) COMPARE_SCRIPT="$2"; shift 2 ;;
    --hybrid-min-throughput-pct) HYBRID_MIN_THROUGHPUT_PCT="$2"; shift 2 ;;
    --hybrid-max-latency-regress-pct) HYBRID_MAX_LATENCY_REGRESS_PCT="$2"; shift 2 ;;
    --hybrid-max-error-regress-abs) HYBRID_MAX_ERROR_REGRESS_ABS="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

for cmd in awk jq grep sed date mktemp mkdir basename dirname; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 127
  fi
done

case "${RUNTIME_TAG_GATE_MODE}" in
  auto|on|off) ;;
  *) echo "Invalid --runtime-tag-gate: ${RUNTIME_TAG_GATE_MODE}" >&2; exit 2 ;;
esac

if [[ ! -x "${BENCH_SCRIPT}" ]]; then
  echo "Benchmark script not executable: ${BENCH_SCRIPT}" >&2
  exit 2
fi
if [[ ! -x "${COMPARE_SCRIPT}" ]]; then
  echo "Compare script not executable: ${COMPARE_SCRIPT}" >&2
  exit 2
fi

RUNTIME_TAG_GATE_EFFECTIVE="false"
case "${RUNTIME_TAG_GATE_MODE}" in
  on) RUNTIME_TAG_GATE_EFFECTIVE="true" ;;
  off) RUNTIME_TAG_GATE_EFFECTIVE="false" ;;
  auto)
    if [[ -n "${HEALTH_URL}" ]]; then
      RUNTIME_TAG_GATE_EFFECTIVE="true"
    fi
    ;;
esac

is_number() {
  [[ "${1}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]
}

timestamp="$(date +%Y%m%d-%H%M%S)"
if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="matrix-${timestamp}"
fi

matrix_dir="${OUT_DIR}/${RUN_NAME}"
runs_root="${matrix_dir}/runs"
compare_root="${matrix_dir}/compare"
mkdir -p "${runs_root}" "${compare_root}"

run_profile() {
  local profile="$1"
  local log_file="${matrix_dir}/run-${profile}.log"
  local run_out_root="${runs_root}/${profile}"
  mkdir -p "${run_out_root}"
  : > "${log_file}"

  local -a cmd
  cmd=(
    "${BENCH_SCRIPT}"
    --endpoint "${ENDPOINT}"
    --model "${MODEL_ID}"
    --requests "${REQUESTS}"
    --concurrency "${CONCURRENCY}"
    --adapters "${ADAPTERS}"
    --zipf-alpha "${ZIPF_ALPHA}"
    --mix "${MIX}"
    --seed "${SEED}"
    --api-key-header "${API_KEY_HEADER}"
    --adapter-param-key "${ADAPTER_PARAM_KEY}"
    --advanced-profile "${profile}"
    --telemetry "${TELEMETRY_MODE}"
    --gpu-telemetry "${GPU_TELEMETRY_MODE}"
    --sample-interval-sec "${SAMPLE_INTERVAL_SEC}"
    --out-dir "${run_out_root}"
  )
  if [[ -n "${API_KEY}" ]]; then
    cmd+=(--api-key "${API_KEY}")
  fi
  if [[ -n "${HEALTH_URL}" ]]; then
    cmd+=(--health-url "${HEALTH_URL}")
  fi

  "${cmd[@]}" | tee -a "${log_file}" >/dev/null

  local artifact_dir
  artifact_dir="$(grep -E '^Artifacts:' "${log_file}" | tail -n1 | sed -E 's/^Artifacts:[[:space:]]*//')"
  if [[ -z "${artifact_dir}" || ! -d "${artifact_dir}" ]]; then
    echo "Failed to resolve artifact directory for profile ${profile}" >&2
    exit 2
  fi
  echo "${artifact_dir}"
}

to_num() {
  local v="$1"
  if is_number "${v}"; then
    echo "${v}"
  else
    echo "0"
  fi
}

status_for() {
  if [[ "$1" == "true" ]]; then
    echo "PASS"
  else
    echo "FAIL"
  fi
}

baseline_run="$(run_profile baseline)"
hybrid_run="$(run_profile hybrid-fp8-bf16)"
sa2_run="$(run_profile sageattention2-intent)"

"${COMPARE_SCRIPT}" \
  --baseline "${baseline_run}" \
  --candidate "${hybrid_run}" \
  --candidate "${sa2_run}" \
  --out-dir "${compare_root}" \
  --name "baseline-vs-candidates" > "${matrix_dir}/compare.log"

compare_json="${compare_root}/baseline-vs-candidates/report.json"
if [[ ! -f "${compare_json}" ]]; then
  echo "Comparison report missing: ${compare_json}" >&2
  exit 2
fi

hybrid_req_delta_pct="$(jq -r '.comparisons[] | select(.candidate_profile=="hybrid-fp8-bf16" and .metric=="throughput_req_s") | .delta_pct' "${compare_json}" | head -n1)"
hybrid_tokens_delta_pct="$(jq -r '.comparisons[] | select(.candidate_profile=="hybrid-fp8-bf16" and .metric=="throughput_tokens_s_est") | .delta_pct' "${compare_json}" | head -n1)"
hybrid_latency_delta_pct="$(jq -r '.comparisons[] | select(.candidate_profile=="hybrid-fp8-bf16" and .metric=="latency_all_p95_ms") | .delta_pct' "${compare_json}" | head -n1)"
hybrid_error_delta_abs="$(jq -r '.comparisons[] | select(.candidate_profile=="hybrid-fp8-bf16" and .metric=="error_rate") | .delta_abs' "${compare_json}" | head -n1)"

hybrid_req_delta_pct="$(to_num "${hybrid_req_delta_pct}")"
hybrid_tokens_delta_pct="$(to_num "${hybrid_tokens_delta_pct}")"
hybrid_latency_delta_pct="$(to_num "${hybrid_latency_delta_pct}")"
hybrid_error_delta_abs="$(to_num "${hybrid_error_delta_abs}")"

gate_hybrid_throughput="$(awk -v a="${hybrid_req_delta_pct}" -v b="${hybrid_tokens_delta_pct}" -v min="${HYBRID_MIN_THROUGHPUT_PCT}" 'BEGIN { if (a >= min || b >= min) print "true"; else print "false"; }')"
gate_hybrid_latency="$(awk -v d="${hybrid_latency_delta_pct}" -v max="${HYBRID_MAX_LATENCY_REGRESS_PCT}" 'BEGIN { if (d <= max) print "true"; else print "false"; }')"
gate_hybrid_error="$(awk -v d="${hybrid_error_delta_abs}" -v max="${HYBRID_MAX_ERROR_REGRESS_ABS}" 'BEGIN { if (d <= max) print "true"; else print "false"; }')"

sa2_profile_expected="$(jq -r '.advanced_profile // "unknown"' "${sa2_run}/summary.json")"
sa2_expected_sage="$(jq -r '.profile_expected_sage_requested // "unknown"' "${sa2_run}/summary.json")"
sa2_runtime_mode="$(jq -r '.runtime_tags.advanced_attention_mode // "unknown"' "${sa2_run}/summary.json")"
sa2_runtime_requested="$(jq -r '.runtime_tags.advanced_sage_attention2_requested // "unknown"' "${sa2_run}/summary.json")"

gate_sa2_profile="$( [[ "${sa2_profile_expected}" == "sageattention2-intent" ]] && echo "true" || echo "false" )"
gate_sa2_expected_flag="$( [[ "${sa2_expected_sage}" == "true" ]] && echo "true" || echo "false" )"
gate_sa2_runtime_requested_known="$( [[ "${sa2_runtime_requested}" == "unknown" ]] && echo "false" || echo "true" )"
if [[ "${sa2_runtime_requested}" == "unknown" ]]; then
  gate_sa2_runtime_requested_true="false"
else
  gate_sa2_runtime_requested_true="$( [[ "${sa2_runtime_requested}" == "true" ]] && echo "true" || echo "false" )"
fi
if [[ "${sa2_runtime_mode}" == "unknown" ]]; then
  gate_sa2_runtime_baseline="false"
else
  gate_sa2_runtime_baseline="$( [[ "${sa2_runtime_mode}" == "baseline" ]] && echo "true" || echo "false" )"
fi

if [[ "${RUNTIME_TAG_GATE_EFFECTIVE}" == "true" ]]; then
  sa2_runtime_required="true"
  gate_sa2_runtime_required_pass="$(awk \
    -v k="${gate_sa2_runtime_requested_known}" \
    -v r="${gate_sa2_runtime_requested_true}" \
    -v m="${gate_sa2_runtime_baseline}" \
    'BEGIN { if (k=="true" && r=="true" && m=="true") print "true"; else print "false"; }')"
else
  sa2_runtime_required="false"
  gate_sa2_runtime_required_pass="true"
fi

matrix_pass="$(awk \
  -v g1="${gate_hybrid_throughput}" \
  -v g2="${gate_hybrid_latency}" \
  -v g3="${gate_hybrid_error}" \
  -v g4="${gate_sa2_profile}" \
  -v g5="${gate_sa2_expected_flag}" \
  -v g6="${gate_sa2_runtime_required_pass}" \
  'BEGIN { if (g1=="true" && g2=="true" && g3=="true" && g4=="true" && g5=="true" && g6=="true") print "true"; else print "false"; }')"

gate_txt="${matrix_dir}/gate-summary.txt"
gate_json="${matrix_dir}/gate-summary.json"
matrix_json="${matrix_dir}/matrix-summary.json"

{
  echo "matrix_name=${RUN_NAME}"
  echo "generated_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "baseline_run=${baseline_run}"
  echo "hybrid_run=${hybrid_run}"
  echo "sa2_intent_run=${sa2_run}"
  echo
  echo "hybrid_req_delta_pct=${hybrid_req_delta_pct}"
  echo "hybrid_tokens_delta_pct=${hybrid_tokens_delta_pct}"
  echo "hybrid_latency_p95_delta_pct=${hybrid_latency_delta_pct}"
  echo "hybrid_error_delta_abs=${hybrid_error_delta_abs}"
  echo
  echo "gate_hybrid_throughput=$(status_for "${gate_hybrid_throughput}") (min >= ${HYBRID_MIN_THROUGHPUT_PCT}%)"
  echo "gate_hybrid_latency=$(status_for "${gate_hybrid_latency}") (max <= ${HYBRID_MAX_LATENCY_REGRESS_PCT}%)"
  echo "gate_hybrid_error=$(status_for "${gate_hybrid_error}") (max <= ${HYBRID_MAX_ERROR_REGRESS_ABS})"
  echo "gate_sa2_profile_intent=$(status_for "${gate_sa2_profile}")"
  echo "gate_sa2_expected_requested_flag=$(status_for "${gate_sa2_expected_flag}")"
  echo "runtime_tag_gate_mode=${RUNTIME_TAG_GATE_MODE}"
  echo "runtime_tag_gate_effective=${RUNTIME_TAG_GATE_EFFECTIVE}"
  echo "gate_sa2_runtime_requested_known=$(status_for "${gate_sa2_runtime_requested_known}")"
  echo "gate_sa2_runtime_requested_true=$(status_for "${gate_sa2_runtime_requested_true}")"
  echo "gate_sa2_runtime_mode_baseline=$(status_for "${gate_sa2_runtime_baseline}")"
  echo "gate_sa2_runtime_required_pass=$(status_for "${gate_sa2_runtime_required_pass}")"
  echo
  echo "matrix_result=$(status_for "${matrix_pass}")"
} > "${gate_txt}"

jq -n \
  --arg matrix_name "${RUN_NAME}" \
  --arg generated_at "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  --arg baseline_run "${baseline_run}" \
  --arg hybrid_run "${hybrid_run}" \
  --arg sa2_run "${sa2_run}" \
  --arg hybrid_req_delta_pct "${hybrid_req_delta_pct}" \
  --arg hybrid_tokens_delta_pct "${hybrid_tokens_delta_pct}" \
  --arg hybrid_latency_delta_pct "${hybrid_latency_delta_pct}" \
  --arg hybrid_error_delta_abs "${hybrid_error_delta_abs}" \
  --arg hybrid_min_throughput_pct "${HYBRID_MIN_THROUGHPUT_PCT}" \
  --arg hybrid_max_latency_regress_pct "${HYBRID_MAX_LATENCY_REGRESS_PCT}" \
  --arg hybrid_max_error_regress_abs "${HYBRID_MAX_ERROR_REGRESS_ABS}" \
  --arg runtime_tag_gate_mode "${RUNTIME_TAG_GATE_MODE}" \
  --arg runtime_tag_gate_effective "${RUNTIME_TAG_GATE_EFFECTIVE}" \
  --arg g_hybrid_throughput "${gate_hybrid_throughput}" \
  --arg g_hybrid_latency "${gate_hybrid_latency}" \
  --arg g_hybrid_error "${gate_hybrid_error}" \
  --arg g_sa2_profile "${gate_sa2_profile}" \
  --arg g_sa2_expected_flag "${gate_sa2_expected_flag}" \
  --arg g_sa2_runtime_requested_known "${gate_sa2_runtime_requested_known}" \
  --arg g_sa2_runtime_requested_true "${gate_sa2_runtime_requested_true}" \
  --arg g_sa2_runtime_mode_baseline "${gate_sa2_runtime_baseline}" \
  --arg g_sa2_runtime_required_pass "${gate_sa2_runtime_required_pass}" \
  --arg matrix_pass "${matrix_pass}" \
  --arg compare_report "${compare_json}" \
  '{
    matrix_name: $matrix_name,
    generated_at_utc: $generated_at,
    runs: {
      baseline: $baseline_run,
      hybrid_fp8_bf16: $hybrid_run,
      sageattention2_intent: $sa2_run
    },
    thresholds: {
      hybrid_min_throughput_pct: ($hybrid_min_throughput_pct | tonumber),
      hybrid_max_latency_regress_pct: ($hybrid_max_latency_regress_pct | tonumber),
      hybrid_max_error_regress_abs: ($hybrid_max_error_regress_abs | tonumber),
      runtime_tag_gate_mode: $runtime_tag_gate_mode,
      runtime_tag_gate_effective: ($runtime_tag_gate_effective == "true")
    },
    observed: {
      hybrid_req_delta_pct: ($hybrid_req_delta_pct | tonumber),
      hybrid_tokens_delta_pct: ($hybrid_tokens_delta_pct | tonumber),
      hybrid_latency_p95_delta_pct: ($hybrid_latency_delta_pct | tonumber),
      hybrid_error_delta_abs: ($hybrid_error_delta_abs | tonumber)
    },
    gates: {
      hybrid_throughput: ($g_hybrid_throughput == "true"),
      hybrid_latency: ($g_hybrid_latency == "true"),
      hybrid_error: ($g_hybrid_error == "true"),
      sa2_profile_intent: ($g_sa2_profile == "true"),
      sa2_expected_requested_flag: ($g_sa2_expected_flag == "true"),
      sa2_runtime_requested_known: ($g_sa2_runtime_requested_known == "true"),
      sa2_runtime_requested_true: ($g_sa2_runtime_requested_true == "true"),
      sa2_runtime_mode_baseline: ($g_sa2_runtime_mode_baseline == "true"),
      sa2_runtime_required_pass: ($g_sa2_runtime_required_pass == "true")
    },
    matrix_pass: ($matrix_pass == "true"),
    compare_report: $compare_report
  }' > "${gate_json}"

jq -n \
  --arg matrix_name "${RUN_NAME}" \
  --arg generated_at "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  --arg baseline_run "${baseline_run}" \
  --arg hybrid_run "${hybrid_run}" \
  --arg sa2_run "${sa2_run}" \
  --arg compare_report "${compare_json}" \
  --arg gate_summary "${gate_json}" \
  '{
    matrix_name: $matrix_name,
    generated_at_utc: $generated_at,
    runs: {
      baseline: $baseline_run,
      hybrid_fp8_bf16: $hybrid_run,
      sageattention2_intent: $sa2_run
    },
    reports: {
      compare: $compare_report,
      gates: $gate_summary
    }
  }' > "${matrix_json}"

echo "Matrix completed:"
echo "  - ${matrix_json}"
echo "  - ${gate_txt}"
echo "  - ${gate_json}"
if [[ "${matrix_pass}" == "true" ]]; then
  echo "Matrix result: PASS"
else
  echo "Matrix result: FAIL"
  exit 1
fi
