#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bench-multilora-zipf.sh [options]

Generates and runs a Multi-LoRA Zipf workload against Gollek inference API.
Outputs run artifacts under ops/benchmarks/<run_id>/.

Options:
  --endpoint URL            Inference endpoint (default: http://localhost:8080/api/v1/inference)
  --model ID                Model id to request (default: llama-3-8b)
  --requests N              Total requests (default: 200)
  --concurrency N           Parallel workers (default: 16)
  --adapters N              Unique adapter cardinality (default: 128)
  --zipf-alpha F            Zipf alpha (default: 1.0)
  --mix NAME                Workload mix: prefill-heavy|balanced|decode-heavy (default: balanced)
  --api-key KEY             Optional API key value
  --api-key-header NAME     API key header name (default: X-API-Key)
  --adapter-param-key NAME  Adapter parameter key in request payload (default: adapter_id)
  --health-url URL          Optional runtime health URL to capture advanced mode tags
  --out-dir PATH            Output root directory (default: ops/benchmarks)
  --seed N                  Random seed for reproducibility (default: 42)
  --help                    Show this help
USAGE
}

ENDPOINT="http://localhost:8080/api/v1/inference"
MODEL_ID="llama-3-8b"
REQUESTS=200
CONCURRENCY=16
ADAPTERS=128
ZIPF_ALPHA="1.0"
MIX="balanced"
API_KEY=""
API_KEY_HEADER="X-API-Key"
ADAPTER_PARAM_KEY="adapter_id"
HEALTH_URL=""
OUT_DIR="ops/benchmarks"
SEED=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --model) MODEL_ID="$2"; shift 2 ;;
    --requests) REQUESTS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --adapters) ADAPTERS="$2"; shift 2 ;;
    --zipf-alpha) ZIPF_ALPHA="$2"; shift 2 ;;
    --mix) MIX="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --api-key-header) API_KEY_HEADER="$2"; shift 2 ;;
    --adapter-param-key) ADAPTER_PARAM_KEY="$2"; shift 2 ;;
    --health-url) HEALTH_URL="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

case "$MIX" in
  prefill-heavy|balanced|decode-heavy) ;;
  *) echo "Invalid --mix: $MIX" >&2; exit 2 ;;
esac

for cmd in awk sort xargs curl mktemp date; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 127
  fi
done

if [[ -z "${ADAPTER_PARAM_KEY}" ]]; then
  echo "--adapter-param-key must not be empty" >&2
  exit 2
fi

now_iso8601() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

timestamp="$(date +%Y%m%d-%H%M%S)"
run_id="zipf-${MIX}-a${ZIPF_ALPHA}-n${REQUESTS}-c${CONCURRENCY}-${timestamp}"
run_dir="${OUT_DIR}/${run_id}"
mkdir -p "${run_dir}"

plan_file="${run_dir}/plan.txt"
result_file="${run_dir}/results.csv"
meta_file="${run_dir}/meta.env"
config_json_file="${run_dir}/config.json"
summary_json_file="${run_dir}/summary.json"
runtime_tags_file="${run_dir}/runtime-tags.json"
lat_all_file="${run_dir}/latency-all-ms.txt"
lat_ttft_file="${run_dir}/latency-ttft-ms.txt"
lat_tpot_file="${run_dir}/latency-tpot-ms.txt"
status_counts_file="${run_dir}/status-counts.txt"
adapter_switch_file="${run_dir}/adapter-switch.txt"
start_epoch="$(date +%s)"

cat > "${meta_file}" <<META
RUN_ID=${run_id}
ENDPOINT=${ENDPOINT}
MODEL_ID=${MODEL_ID}
REQUESTS=${REQUESTS}
CONCURRENCY=${CONCURRENCY}
ADAPTERS=${ADAPTERS}
ZIPF_ALPHA=${ZIPF_ALPHA}
MIX=${MIX}
SEED=${SEED}
API_KEY_HEADER=${API_KEY_HEADER}
ADAPTER_PARAM_KEY=${ADAPTER_PARAM_KEY}
HEALTH_URL=${HEALTH_URL}
START_TS=$(now_iso8601)
META

cat > "${config_json_file}" <<CONFIG
{
  "run_id": "${run_id}",
  "endpoint": "${ENDPOINT}",
  "model_id": "${MODEL_ID}",
  "requests": ${REQUESTS},
  "concurrency": ${CONCURRENCY},
  "adapters": ${ADAPTERS},
  "zipf_alpha": ${ZIPF_ALPHA},
  "mix": "${MIX}",
  "seed": ${SEED},
  "api_key_header": "${API_KEY_HEADER}",
  "adapter_param_key": "${ADAPTER_PARAM_KEY}",
  "health_url": "${HEALTH_URL}"
}
CONFIG

echo "request_id,phase,adapter_id,http_code,latency_ms,error" > "${result_file}"

generate_plan() {
  awk -v requests="${REQUESTS}" -v adapters="${ADAPTERS}" -v alpha="${ZIPF_ALPHA}" -v mix="${MIX}" -v seed="${SEED}" '
    function phase_for_mix(u, m) {
      if (m == "prefill-heavy") return (u < 0.70 ? "prefill" : "decode");
      if (m == "decode-heavy") return (u < 0.30 ? "prefill" : "decode");
      return (u < 0.50 ? "prefill" : "decode");
    }
    function sample_zipf(n, a,   z, i, u, acc, w) {
      z = 0.0;
      for (i = 1; i <= n; i++) z += 1.0 / (i ^ a);
      u = rand() * z;
      acc = 0.0;
      for (i = 1; i <= n; i++) {
        w = 1.0 / (i ^ a);
        acc += w;
        if (acc >= u) return i;
      }
      return n;
    }
    BEGIN {
      srand(seed);
      for (i = 1; i <= requests; i++) {
        phase = phase_for_mix(rand(), mix);
        idx = sample_zipf(adapters, alpha);
        printf("%d|%s|adapter-%04d\n", i, phase, idx);
      }
    }
  ' > "${plan_file}"
}

build_payload() {
  local request_id="$1"
  local phase="$2"
  local adapter_id="$3"
  local prompt
  local max_tokens
  local temperature

  if [[ "${phase}" == "prefill" ]]; then
    prompt="Summarize this document with high fidelity and preserve key entities. $(printf 'token %.0s' {1..220})"
    max_tokens=96
    temperature=0.2
  else
    prompt="Continue naturally: The quick brown fox"
    max_tokens=256
    temperature=0.8
  fi

  cat <<PAYLOAD
{"model":"${MODEL_ID}","requestId":"zipf-${request_id}","messages":[{"role":"user","content":"${prompt}"}],"parameters":{"${ADAPTER_PARAM_KEY}":"${adapter_id}","max_tokens":${max_tokens},"temperature":${temperature},"benchmark_phase":"${phase}"}}
PAYLOAD
}

run_one() {
  local line="$1"
  local request_id phase adapter_id
  IFS='|' read -r request_id phase adapter_id <<< "${line}"

  local payload tmp_out http_code latency_s latency_ms error_msg
  payload="$(build_payload "${request_id}" "${phase}" "${adapter_id}")"
  tmp_out="$(mktemp)"

  if [[ -n "${API_KEY}" ]]; then
    read -r http_code latency_s < <(
      curl -s -o "${tmp_out}" -w "%{http_code} %{time_total}" \
        -H "Content-Type: application/json" \
        -H "${API_KEY_HEADER}: ${API_KEY}" \
        -d "${payload}" "${ENDPOINT}" 2>/dev/null || echo "000 0"
    )
  else
    read -r http_code latency_s < <(
      curl -s -o "${tmp_out}" -w "%{http_code} %{time_total}" \
        -H "Content-Type: application/json" \
        -d "${payload}" "${ENDPOINT}" 2>/dev/null || echo "000 0"
    )
  fi

  latency_ms="$(awk -v s="${latency_s}" 'BEGIN { printf("%.3f", s * 1000.0) }')"
  error_msg=""
  if [[ "${http_code}" != "200" ]]; then
    error_msg="$(tr '\n' ' ' < "${tmp_out}" | tr ',' ';' | cut -c1-200)"
  fi

  printf "%s,%s,%s,%s,%s,%s\n" \
    "${request_id}" "${phase}" "${adapter_id}" "${http_code}" "${latency_ms}" "${error_msg}" >> "${result_file}"
  rm -f "${tmp_out}"
}

percentile_from_file() {
  local file="$1"
  local pct="$2"
  awk -v pct="${pct}" '
    { vals[++n] = $1 + 0.0 }
    END {
      if (n == 0) {
        printf("0.000\n");
        exit 0;
      }
      idx = int(((pct / 100.0) * n) + 0.999999);
      if (idx < 1) idx = 1;
      if (idx > n) idx = n;
      printf("%.3f\n", vals[idx]);
    }
  ' "${file}"
}

summarize_csv() {
  local summary_file="${run_dir}/summary.txt"
  local end_epoch duration_seconds duration_for_rate
  local total ok fail prefill decode
  local p50_all p95_all p50_ttft p95_ttft p50_tpot p95_tpot
  local req_s tokens_s estimated_tokens error_rate adapter_switches adapter_switch_ratio

  awk -F',' '
    NR==1 { next }
    {
      total++;
      if ($4 == "200") ok++;
      if ($4 != "200") fail++;
      if ($2 == "prefill") prefill++;
      if ($2 == "decode") decode++;
      print $5 > "'"${lat_all_file}"'";
      if ($2 == "prefill") print $5 > "'"${lat_ttft_file}"'";
      if ($2 == "decode") print $5 > "'"${lat_tpot_file}"'";
      status[$4]++;
    }
    END {
      printf("%d %d %d %d %d\n", total, ok, fail, prefill, decode);
      for (code in status) {
        printf("%s %d\n", code, status[code]) > "'"${status_counts_file}"'";
      }
    }
  ' "${result_file}" > "${run_dir}/counts.txt"

  read -r total ok fail prefill decode < "${run_dir}/counts.txt"

  : > "${lat_all_file}"
  : > "${lat_ttft_file}"
  : > "${lat_tpot_file}"
  awk -F',' 'NR>1 { print $5 > "'"${lat_all_file}"'"; if ($2=="prefill") print $5 > "'"${lat_ttft_file}"'"; if ($2=="decode") print $5 > "'"${lat_tpot_file}"'"; }' "${result_file}"

  sort -n "${lat_all_file}" -o "${lat_all_file}" || true
  sort -n "${lat_ttft_file}" -o "${lat_ttft_file}" || true
  sort -n "${lat_tpot_file}" -o "${lat_tpot_file}" || true

  p50_all="$(percentile_from_file "${lat_all_file}" 50)"
  p95_all="$(percentile_from_file "${lat_all_file}" 95)"
  p50_ttft="$(percentile_from_file "${lat_ttft_file}" 50)"
  p95_ttft="$(percentile_from_file "${lat_ttft_file}" 95)"
  p50_tpot="$(percentile_from_file "${lat_tpot_file}" 50)"
  p95_tpot="$(percentile_from_file "${lat_tpot_file}" 95)"

  end_epoch="$(date +%s)"
  duration_seconds=$(( end_epoch - start_epoch ))
  if (( duration_seconds <= 0 )); then
    duration_for_rate="1"
  else
    duration_for_rate="${duration_seconds}"
  fi

  req_s="$(awk -v t="${total:-0}" -v d="${duration_for_rate}" 'BEGIN { printf("%.3f", t/d) }')"
  estimated_tokens=$(( (prefill * 96) + (decode * 256) ))
  tokens_s="$(awk -v t="${estimated_tokens}" -v d="${duration_for_rate}" 'BEGIN { printf("%.3f", t/d) }')"
  error_rate="$(awk -v f="${fail:-0}" -v t="${total:-1}" 'BEGIN { if (t <= 0) printf("0.0000"); else printf("%.4f", f/t) }')"

  awk -F'|' '
    { if (NR > 1 && $3 != prev) sw++; prev = $3; n++; }
    END {
      if (n <= 1) {
        printf("0 0.0000\n");
      } else {
        ratio = sw / (n - 1);
        printf("%d %.4f\n", sw, ratio);
      }
    }
  ' "${plan_file}" > "${adapter_switch_file}"
  read -r adapter_switches adapter_switch_ratio < "${adapter_switch_file}"

  cat > "${summary_file}" <<SUMMARY

total=${total}
ok=${ok}
fail=${fail}
error_rate=${error_rate}
prefill=${prefill}
decode=${decode}
duration_seconds=${duration_seconds}
throughput_req_s=${req_s}
throughput_tokens_s_est=${tokens_s}
latency_all_p50_ms=${p50_all}
latency_all_p95_ms=${p95_all}
ttft_p50_ms=${p50_ttft}
ttft_p95_ms=${p95_ttft}
tpot_p50_ms=${p50_tpot}
tpot_p95_ms=${p95_tpot}
adapter_switches=${adapter_switches}
adapter_switch_ratio=${adapter_switch_ratio}
SUMMARY
  cat "${summary_file}"

  cat > "${summary_json_file}" <<JSON
{
  "run_id": "${run_id}",
  "total": ${total},
  "ok": ${ok},
  "fail": ${fail},
  "error_rate": ${error_rate},
  "prefill": ${prefill},
  "decode": ${decode},
  "duration_seconds": ${duration_seconds},
  "throughput_req_s": ${req_s},
  "throughput_tokens_s_est": ${tokens_s},
  "latency_all_p50_ms": ${p50_all},
  "latency_all_p95_ms": ${p95_all},
  "ttft_p50_ms": ${p50_ttft},
  "ttft_p95_ms": ${p95_ttft},
  "tpot_p50_ms": ${p50_tpot},
  "tpot_p95_ms": ${p95_tpot},
  "adapter_switches": ${adapter_switches},
  "adapter_switch_ratio": ${adapter_switch_ratio}
}
JSON
}

capture_runtime_tags() {
  if [[ -z "${HEALTH_URL}" ]]; then
    return
  fi

  local tmp_health enabled mode reason detected_sm
  local sage_requested sage_active sage_reason
  local rowwise_active rowwise_reason rowwise_scale_count rowwise_scale_mean rowwise_calibration_source
  tmp_health="$(mktemp)"
  if [[ -n "${API_KEY}" ]]; then
    curl -s -H "${API_KEY_HEADER}: ${API_KEY}" "${HEALTH_URL}" -o "${tmp_health}" 2>/dev/null || true
  else
    curl -s "${HEALTH_URL}" -o "${tmp_health}" 2>/dev/null || true
  fi

  if [[ ! -s "${tmp_health}" ]]; then
    echo "{\"health_url\":\"${HEALTH_URL}\",\"status\":\"unavailable\"}" > "${runtime_tags_file}"
    rm -f "${tmp_health}"
    return
  fi

  if command -v jq >/dev/null 2>&1; then
    enabled="$(jq -r '.. | .advanced_effective_enabled? // empty' "${tmp_health}" | head -n1)"
    mode="$(jq -r '.. | .advanced_attention_mode? // empty' "${tmp_health}" | head -n1)"
    reason="$(jq -r '.. | .advanced_reason? // empty' "${tmp_health}" | head -n1)"
    detected_sm="$(jq -r '.. | .advanced_detected_gpu_sm? // empty' "${tmp_health}" | head -n1)"
    sage_requested="$(jq -r '.. | .advanced_sage_attention2_requested? // empty' "${tmp_health}" | head -n1)"
    sage_active="$(jq -r '.. | .advanced_sage_attention2_active? // empty' "${tmp_health}" | head -n1)"
    sage_reason="$(jq -r '.. | .advanced_sage_attention2_reason? // empty' "${tmp_health}" | head -n1)"
    rowwise_active="$(jq -r '.. | .advanced_fp8_rowwise_active? // empty' "${tmp_health}" | head -n1)"
    rowwise_reason="$(jq -r '.. | .advanced_fp8_rowwise_reason? // empty' "${tmp_health}" | head -n1)"
    rowwise_scale_count="$(jq -r '.. | .advanced_fp8_rowwise_scale_count? // empty' "${tmp_health}" | head -n1)"
    rowwise_scale_mean="$(jq -r '.. | .advanced_fp8_rowwise_scale_mean? // empty' "${tmp_health}" | head -n1)"
    rowwise_calibration_source="$(jq -r '.. | .advanced_fp8_rowwise_calibration_source? // empty' "${tmp_health}" | head -n1)"
  else
    enabled="$(grep -o '"advanced_effective_enabled"[[:space:]]*:[[:space:]]*[^,}]*' "${tmp_health}" | head -n1 | awk -F: '{gsub(/[\" ]/, "", $2); print $2}')"
    mode="$(grep -o '"advanced_attention_mode"[[:space:]]*:[[:space:]]*"[^"]*"' "${tmp_health}" | head -n1 | cut -d'"' -f4)"
    reason="$(grep -o '"advanced_reason"[[:space:]]*:[[:space:]]*"[^"]*"' "${tmp_health}" | head -n1 | cut -d'"' -f4)"
    detected_sm="$(grep -o '"advanced_detected_gpu_sm"[[:space:]]*:[[:space:]]*"[^"]*"' "${tmp_health}" | head -n1 | cut -d'"' -f4)"
    sage_requested="$(grep -o '"advanced_sage_attention2_requested"[[:space:]]*:[[:space:]]*[^,}]*' "${tmp_health}" | head -n1 | awk -F: '{gsub(/[\" ]/, "", $2); print $2}')"
    sage_active="$(grep -o '"advanced_sage_attention2_active"[[:space:]]*:[[:space:]]*[^,}]*' "${tmp_health}" | head -n1 | awk -F: '{gsub(/[\" ]/, "", $2); print $2}')"
    sage_reason="$(grep -o '"advanced_sage_attention2_reason"[[:space:]]*:[[:space:]]*"[^"]*"' "${tmp_health}" | head -n1 | cut -d'"' -f4)"
    rowwise_active="$(grep -o '"advanced_fp8_rowwise_active"[[:space:]]*:[[:space:]]*[^,}]*' "${tmp_health}" | head -n1 | awk -F: '{gsub(/[\" ]/, "", $2); print $2}')"
    rowwise_reason="$(grep -o '"advanced_fp8_rowwise_reason"[[:space:]]*:[[:space:]]*"[^"]*"' "${tmp_health}" | head -n1 | cut -d'"' -f4)"
    rowwise_scale_count="$(grep -o '"advanced_fp8_rowwise_scale_count"[[:space:]]*:[[:space:]]*[^,}]*' "${tmp_health}" | head -n1 | awk -F: '{gsub(/[\" ]/, "", $2); print $2}')"
    rowwise_scale_mean="$(grep -o '"advanced_fp8_rowwise_scale_mean"[[:space:]]*:[[:space:]]*[^,}]*' "${tmp_health}" | head -n1 | awk -F: '{gsub(/[\" ]/, "", $2); print $2}')"
    rowwise_calibration_source="$(grep -o '"advanced_fp8_rowwise_calibration_source"[[:space:]]*:[[:space:]]*"[^"]*"' "${tmp_health}" | head -n1 | cut -d'"' -f4)"
  fi

  enabled="${enabled:-unknown}"
  mode="${mode:-unknown}"
  reason="${reason:-unknown}"
  detected_sm="${detected_sm:-unknown}"
  sage_requested="${sage_requested:-unknown}"
  sage_active="${sage_active:-unknown}"
  sage_reason="${sage_reason:-unknown}"
  rowwise_active="${rowwise_active:-unknown}"
  rowwise_reason="${rowwise_reason:-unknown}"
  rowwise_scale_count="${rowwise_scale_count:-unknown}"
  rowwise_scale_mean="${rowwise_scale_mean:-unknown}"
  rowwise_calibration_source="${rowwise_calibration_source:-unknown}"

  cat > "${runtime_tags_file}" <<RUNTIME
{
  "health_url": "${HEALTH_URL}",
  "advanced_effective_enabled": "${enabled}",
  "advanced_attention_mode": "${mode}",
  "advanced_reason": "${reason}",
  "advanced_detected_gpu_sm": "${detected_sm}",
  "advanced_sage_attention2_requested": "${sage_requested}",
  "advanced_sage_attention2_active": "${sage_active}",
  "advanced_sage_attention2_reason": "${sage_reason}",
  "advanced_fp8_rowwise_active": "${rowwise_active}",
  "advanced_fp8_rowwise_reason": "${rowwise_reason}",
  "advanced_fp8_rowwise_scale_count": "${rowwise_scale_count}",
  "advanced_fp8_rowwise_scale_mean": "${rowwise_scale_mean}",
  "advanced_fp8_rowwise_calibration_source": "${rowwise_calibration_source}"
}
RUNTIME

  rm -f "${tmp_health}"
}

merge_runtime_tags_into_summary() {
  if [[ ! -f "${runtime_tags_file}" ]]; then
    return
  fi
  if command -v jq >/dev/null 2>&1; then
    local merged_file="${run_dir}/summary.merged.json"
    jq --slurpfile rt "${runtime_tags_file}" '. + {runtime_tags: $rt[0]}' "${summary_json_file}" > "${merged_file}" \
      && mv "${merged_file}" "${summary_json_file}"
  fi
}

export -f build_payload
export -f run_one
export MODEL_ID ENDPOINT API_KEY API_KEY_HEADER ADAPTER_PARAM_KEY result_file

generate_plan
cat "${plan_file}" | xargs -I{} -P "${CONCURRENCY}" bash -lc 'run_one "$@"' _ "{}"

echo "END_TS=$(now_iso8601)" >> "${meta_file}"
summarize_csv
capture_runtime_tags
merge_runtime_tags_into_summary
echo "Artifacts: ${run_dir}"
