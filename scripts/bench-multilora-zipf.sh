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
  --advanced-profile NAME   benchmark profile: baseline|hybrid-fp8-bf16|sageattention2-intent (default: baseline)
  --telemetry MODE          Host telemetry mode: auto|on|off (default: auto)
  --gpu-telemetry MODE      GPU telemetry mode: auto|on|off (default: auto)
  --sample-interval-sec N   Telemetry sampling interval in seconds (default: 1)
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
ADVANCED_PROFILE="baseline"
TELEMETRY_MODE="auto"
GPU_TELEMETRY_MODE="auto"
SAMPLE_INTERVAL_SEC=1
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
    --advanced-profile) ADVANCED_PROFILE="$2"; shift 2 ;;
    --telemetry) TELEMETRY_MODE="$2"; shift 2 ;;
    --gpu-telemetry) GPU_TELEMETRY_MODE="$2"; shift 2 ;;
    --sample-interval-sec) SAMPLE_INTERVAL_SEC="$2"; shift 2 ;;
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

case "$ADVANCED_PROFILE" in
  baseline|hybrid-fp8-bf16|sageattention2-intent) ;;
  *) echo "Invalid --advanced-profile: $ADVANCED_PROFILE" >&2; exit 2 ;;
esac

case "$TELEMETRY_MODE" in
  auto|on|off) ;;
  *) echo "Invalid --telemetry: $TELEMETRY_MODE" >&2; exit 2 ;;
esac

case "$GPU_TELEMETRY_MODE" in
  auto|on|off) ;;
  *) echo "Invalid --gpu-telemetry: $GPU_TELEMETRY_MODE" >&2; exit 2 ;;
esac

if ! [[ "${SAMPLE_INTERVAL_SEC}" =~ ^[0-9]+$ ]] || (( SAMPLE_INTERVAL_SEC <= 0 )); then
  echo "Invalid --sample-interval-sec: ${SAMPLE_INTERVAL_SEC}" >&2
  exit 2
fi

for cmd in awk sed sort xargs curl mktemp date; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 127
  fi
done

if [[ -z "${ADAPTER_PARAM_KEY}" ]]; then
  echo "--adapter-param-key must not be empty" >&2
  exit 2
fi

PROFILE_EXPECTED_MODE="baseline"
PROFILE_EXPECTED_SAGE_REQUESTED="false"
PROFILE_EXPECTED_SAGE_ACTIVE="false"
case "${ADVANCED_PROFILE}" in
  baseline)
    PROFILE_EXPECTED_MODE="baseline"
    ;;
  hybrid-fp8-bf16)
    PROFILE_EXPECTED_MODE="hybrid_fp8_bf16"
    ;;
  sageattention2-intent)
    # SA2 is rollback-only today; this profile tracks rollout intent vs effective mode.
    PROFILE_EXPECTED_MODE="baseline"
    PROFILE_EXPECTED_SAGE_REQUESTED="true"
    ;;
esac

now_iso8601() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

timestamp="$(date +%Y%m%d-%H%M%S)"
run_id="zipf-${ADVANCED_PROFILE}-${MIX}-a${ZIPF_ALPHA}-n${REQUESTS}-c${CONCURRENCY}-${timestamp}"
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
telemetry_csv_file="${run_dir}/telemetry.csv"
telemetry_summary_file="${run_dir}/telemetry-summary.txt"
start_epoch="$(date +%s)"
telemetry_pid=""
telemetry_enabled="false"
gpu_telemetry_enabled="false"

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
ADVANCED_PROFILE=${ADVANCED_PROFILE}
PROFILE_EXPECTED_MODE=${PROFILE_EXPECTED_MODE}
PROFILE_EXPECTED_SAGE_REQUESTED=${PROFILE_EXPECTED_SAGE_REQUESTED}
PROFILE_EXPECTED_SAGE_ACTIVE=${PROFILE_EXPECTED_SAGE_ACTIVE}
TELEMETRY_MODE=${TELEMETRY_MODE}
GPU_TELEMETRY_MODE=${GPU_TELEMETRY_MODE}
SAMPLE_INTERVAL_SEC=${SAMPLE_INTERVAL_SEC}
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
  "health_url": "${HEALTH_URL}",
  "advanced_profile": "${ADVANCED_PROFILE}",
  "profile_expected_mode": "${PROFILE_EXPECTED_MODE}",
  "profile_expected_sage_requested": ${PROFILE_EXPECTED_SAGE_REQUESTED},
  "profile_expected_sage_active": ${PROFILE_EXPECTED_SAGE_ACTIVE},
  "telemetry_mode": "${TELEMETRY_MODE}",
  "gpu_telemetry_mode": "${GPU_TELEMETRY_MODE}",
  "sample_interval_sec": ${SAMPLE_INTERVAL_SEC}
}
CONFIG

echo "request_id,phase,adapter_id,benchmark_profile,http_code,latency_ms,error" > "${result_file}"

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
{"model":"${MODEL_ID}","requestId":"zipf-${ADVANCED_PROFILE}-${request_id}","messages":[{"role":"user","content":"${prompt}"}],"parameters":{"${ADAPTER_PARAM_KEY}":"${adapter_id}","max_tokens":${max_tokens},"temperature":${temperature},"benchmark_phase":"${phase}","benchmark_profile":"${ADVANCED_PROFILE}","benchmark_expected_attention_mode":"${PROFILE_EXPECTED_MODE}","benchmark_expected_sage_attention2_requested":${PROFILE_EXPECTED_SAGE_REQUESTED},"benchmark_expected_sage_attention2_active":${PROFILE_EXPECTED_SAGE_ACTIVE}}}
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

  printf "%s,%s,%s,%s,%s,%s,%s\n" \
    "${request_id}" "${phase}" "${adapter_id}" "${ADVANCED_PROFILE}" "${http_code}" "${latency_ms}" "${error_msg}" >> "${result_file}"
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

detect_telemetry_modes() {
  if [[ "${TELEMETRY_MODE}" == "on" ]]; then
    telemetry_enabled="true"
  elif [[ "${TELEMETRY_MODE}" == "off" ]]; then
    telemetry_enabled="false"
  else
    telemetry_enabled="true"
  fi

  if [[ "${GPU_TELEMETRY_MODE}" == "on" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu_telemetry_enabled="true"
    else
      echo "Warning: --gpu-telemetry=on but nvidia-smi not found; GPU telemetry disabled" >&2
      gpu_telemetry_enabled="false"
    fi
  elif [[ "${GPU_TELEMETRY_MODE}" == "off" ]]; then
    gpu_telemetry_enabled="false"
  else
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu_telemetry_enabled="true"
    else
      gpu_telemetry_enabled="false"
    fi
  fi
}

read_load_avg() {
  local uptime_out normalized
  uptime_out="$(uptime 2>/dev/null || true)"
  if [[ -z "${uptime_out}" ]]; then
    echo ",,"
    return
  fi
  normalized="$(echo "${uptime_out}" | sed -E 's/,//g')"
  awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i == "average:" || $i == "averages:") {
          l1 = $(i+1);
          l5 = $(i+2);
          l15 = $(i+3);
          gsub(/[^0-9.]/, "", l1);
          gsub(/[^0-9.]/, "", l5);
          gsub(/[^0-9.]/, "", l15);
          printf("%s,%s,%s\n", l1, l5, l15);
          exit;
        }
      }
      print ",,";
    }
  ' <<< "${normalized}"
}

read_memory_mb() {
  local total_mb used_mb
  if [[ -r /proc/meminfo ]]; then
    awk '
      /^MemTotal:/ { total = $2 / 1024.0 }
      /^MemAvailable:/ { avail = $2 / 1024.0 }
      END {
        if (total > 0) {
          used = total - avail;
          printf("%.1f,%.1f\n", used, total);
        } else {
          print ",";
        }
      }
    ' /proc/meminfo
    return
  fi

  if command -v sysctl >/dev/null 2>&1 && command -v vm_stat >/dev/null 2>&1; then
    total_mb="$(sysctl -n hw.memsize 2>/dev/null | awk '{printf("%.1f", $1/1024.0/1024.0)}')"
    used_mb="$(vm_stat 2>/dev/null | awk '
      /page size of/ { gsub(/\./, "", $8); page = $8 + 0; if (page == 0) page = 4096; }
      /^Pages active:/ { gsub(/\./, "", $3); active = $3 + 0; }
      /^Pages wired down:/ { gsub(/\./, "", $4); wired = $4 + 0; }
      /^Pages occupied by compressor:/ { gsub(/\./, "", $5); compressed = $5 + 0; }
      END {
        if (page == 0) page = 4096;
        used_bytes = (active + wired + compressed) * page;
        printf("%.1f\n", used_bytes/1024.0/1024.0);
      }
    ')"
    if [[ -n "${used_mb}" && -n "${total_mb}" ]]; then
      echo "${used_mb},${total_mb}"
      return
    fi
  fi

  echo ","
}

read_gpu_metrics() {
  if [[ "${gpu_telemetry_enabled}" != "true" ]]; then
    echo ",,"
    return
  fi

  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | awk -F',' '
    {
      gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
      util_sum += $1;
      mem_used_sum += $2;
      mem_total_sum += $3;
      n++;
    }
    END {
      if (n > 0) {
        printf("%.1f,%.1f,%.1f\n", util_sum / n, mem_used_sum, mem_total_sum);
      } else {
        print ",,";
      }
    }
  '
}

capture_telemetry_sample() {
  local ts load_vals mem_vals gpu_vals
  ts="$(date +%s)"
  load_vals="$(read_load_avg)"
  mem_vals="$(read_memory_mb)"
  gpu_vals="$(read_gpu_metrics)"
  printf "%s,%s,%s,%s\n" "${ts}" "${load_vals}" "${mem_vals}" "${gpu_vals}" >> "${telemetry_csv_file}"
}

start_telemetry_sampler() {
  detect_telemetry_modes
  if [[ "${telemetry_enabled}" != "true" ]]; then
    return
  fi

  echo "timestamp_epoch,host_load_1,host_load_5,host_load_15,host_mem_used_mb,host_mem_total_mb,gpu_util_percent,gpu_mem_used_mb,gpu_mem_total_mb" > "${telemetry_csv_file}"
  (
    while true; do
      capture_telemetry_sample || true
      sleep "${SAMPLE_INTERVAL_SEC}"
    done
  ) &
  telemetry_pid="$!"
}

stop_telemetry_sampler() {
  if [[ -n "${telemetry_pid}" ]]; then
    kill "${telemetry_pid}" >/dev/null 2>&1 || true
    wait "${telemetry_pid}" 2>/dev/null || true
  fi
}

summarize_telemetry() {
  if [[ ! -f "${telemetry_csv_file}" ]]; then
    return
  fi

  awk -F',' '
    NR == 1 { next }
    function isnum(v) { return (v ~ /^-?[0-9]+([.][0-9]+)?$/) }
    {
      if (isnum($2)) { l1_sum += $2; l1_n++; }
      if (isnum($5)) {
        mem_used_sum += $5;
        mem_used_n++;
        if ($5 > mem_used_peak) mem_used_peak = $5;
      }
      if (isnum($8)) {
        gpu_used_n++;
        if ($8 > gpu_used_peak) gpu_used_peak = $8;
      }
      if (isnum($7)) {
        gpu_util[++gpu_n] = $7 + 0.0;
        gpu_util_sum += $7;
      }
    }
    END {
      if (l1_n > 0) printf("host_load_1_avg=%.3f\n", l1_sum / l1_n); else print "host_load_1_avg=unknown";
      if (mem_used_n > 0) {
        printf("host_mem_used_mb_avg=%.1f\n", mem_used_sum / mem_used_n);
        printf("host_mem_used_mb_peak=%.1f\n", mem_used_peak);
      } else {
        print "host_mem_used_mb_avg=unknown";
        print "host_mem_used_mb_peak=unknown";
      }
      if (gpu_n > 0) {
        for (i = 1; i <= gpu_n; i++) {
          for (j = i + 1; j <= gpu_n; j++) {
            if (gpu_util[i] > gpu_util[j]) {
              tmp = gpu_util[i];
              gpu_util[i] = gpu_util[j];
              gpu_util[j] = tmp;
            }
          }
        }
        idx = int((0.95 * gpu_n) + 0.999999);
        if (idx < 1) idx = 1;
        if (idx > gpu_n) idx = gpu_n;
        printf("gpu_util_avg=%.1f\n", gpu_util_sum / gpu_n);
        printf("gpu_util_p95=%.1f\n", gpu_util[idx]);
      } else {
        print "gpu_util_avg=unknown";
        print "gpu_util_p95=unknown";
      }
      if (gpu_used_n > 0) printf("gpu_mem_used_mb_peak=%.1f\n", gpu_used_peak);
      else print "gpu_mem_used_mb_peak=unknown";
    }
  ' "${telemetry_csv_file}" > "${telemetry_summary_file}"
}

summarize_csv() {
  local summary_file="${run_dir}/summary.txt"
  local end_epoch duration_seconds duration_for_rate
  local total ok fail prefill decode
  local p50_all p95_all p50_ttft p95_ttft p50_tpot p95_tpot
  local req_s tokens_s estimated_tokens error_rate adapter_switches adapter_switch_ratio
  local host_load_1_avg host_mem_used_mb_avg host_mem_used_mb_peak
  local gpu_util_avg gpu_util_p95 gpu_mem_used_mb_peak

  awk -F',' '
    NR==1 { next }
    {
      total++;
      if ($5 == "200") ok++;
      if ($5 != "200") fail++;
      if ($2 == "prefill") prefill++;
      if ($2 == "decode") decode++;
      print $6 > "'"${lat_all_file}"'";
      if ($2 == "prefill") print $6 > "'"${lat_ttft_file}"'";
      if ($2 == "decode") print $6 > "'"${lat_tpot_file}"'";
      status[$5]++;
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
  awk -F',' 'NR>1 { print $6 > "'"${lat_all_file}"'"; if ($2=="prefill") print $6 > "'"${lat_ttft_file}"'"; if ($2=="decode") print $6 > "'"${lat_tpot_file}"'"; }' "${result_file}"

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

  host_load_1_avg="unknown"
  host_mem_used_mb_avg="unknown"
  host_mem_used_mb_peak="unknown"
  gpu_util_avg="unknown"
  gpu_util_p95="unknown"
  gpu_mem_used_mb_peak="unknown"
  if [[ -f "${telemetry_summary_file}" ]]; then
    host_load_1_avg="$(awk -F= '$1=="host_load_1_avg"{print $2}' "${telemetry_summary_file}")"
    host_mem_used_mb_avg="$(awk -F= '$1=="host_mem_used_mb_avg"{print $2}' "${telemetry_summary_file}")"
    host_mem_used_mb_peak="$(awk -F= '$1=="host_mem_used_mb_peak"{print $2}' "${telemetry_summary_file}")"
    gpu_util_avg="$(awk -F= '$1=="gpu_util_avg"{print $2}' "${telemetry_summary_file}")"
    gpu_util_p95="$(awk -F= '$1=="gpu_util_p95"{print $2}' "${telemetry_summary_file}")"
    gpu_mem_used_mb_peak="$(awk -F= '$1=="gpu_mem_used_mb_peak"{print $2}' "${telemetry_summary_file}")"
  fi

  cat > "${summary_file}" <<SUMMARY

total=${total}
ok=${ok}
fail=${fail}
error_rate=${error_rate}
advanced_profile=${ADVANCED_PROFILE}
profile_expected_mode=${PROFILE_EXPECTED_MODE}
profile_expected_sage_requested=${PROFILE_EXPECTED_SAGE_REQUESTED}
profile_expected_sage_active=${PROFILE_EXPECTED_SAGE_ACTIVE}
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
host_load_1_avg=${host_load_1_avg}
host_mem_used_mb_avg=${host_mem_used_mb_avg}
host_mem_used_mb_peak=${host_mem_used_mb_peak}
gpu_util_avg=${gpu_util_avg}
gpu_util_p95=${gpu_util_p95}
gpu_mem_used_mb_peak=${gpu_mem_used_mb_peak}
SUMMARY
  cat "${summary_file}"

  cat > "${summary_json_file}" <<JSON
{
  "run_id": "${run_id}",
  "advanced_profile": "${ADVANCED_PROFILE}",
  "profile_expected_mode": "${PROFILE_EXPECTED_MODE}",
  "profile_expected_sage_requested": ${PROFILE_EXPECTED_SAGE_REQUESTED},
  "profile_expected_sage_active": ${PROFILE_EXPECTED_SAGE_ACTIVE},
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
  "adapter_switch_ratio": ${adapter_switch_ratio},
  "host_load_1_avg": "${host_load_1_avg}",
  "host_mem_used_mb_avg": "${host_mem_used_mb_avg}",
  "host_mem_used_mb_peak": "${host_mem_used_mb_peak}",
  "gpu_util_avg": "${gpu_util_avg}",
  "gpu_util_p95": "${gpu_util_p95}",
  "gpu_mem_used_mb_peak": "${gpu_mem_used_mb_peak}"
}
JSON
}

capture_runtime_tags() {
  if [[ -z "${HEALTH_URL}" ]]; then
    return
  fi

  local tmp_health enabled mode reason detected_sm
  local sage_requested sage_active sage_reason
  local rowwise_requested rowwise_active rowwise_reason rowwise_scale_count rowwise_scale_mean rowwise_calibration_source
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
    rowwise_requested="$(jq -r '.. | .advanced_fp8_rowwise_requested? // empty' "${tmp_health}" | head -n1)"
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
    rowwise_requested="$(grep -o '"advanced_fp8_rowwise_requested"[[:space:]]*:[[:space:]]*[^,}]*' "${tmp_health}" | head -n1 | awk -F: '{gsub(/[\" ]/, "", $2); print $2}')"
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
  if [[ -z "${rowwise_requested:-}" ]]; then
    # Backward compatibility with older health payloads
    rowwise_requested="$(grep -o '"advanced_fp8_rowwise_enabled"[[:space:]]*:[[:space:]]*[^,}]*' "${tmp_health}" | head -n1 | awk -F: '{gsub(/[\" ]/, "", $2); print $2}')"
  fi

  rowwise_requested="${rowwise_requested:-unknown}"
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
  "advanced_fp8_rowwise_requested": "${rowwise_requested}",
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
export ADVANCED_PROFILE PROFILE_EXPECTED_MODE PROFILE_EXPECTED_SAGE_REQUESTED PROFILE_EXPECTED_SAGE_ACTIVE

generate_plan
start_telemetry_sampler
trap 'stop_telemetry_sampler' EXIT
cat "${plan_file}" | xargs -I{} -P "${CONCURRENCY}" bash -lc 'run_one "$@"' _ "{}"
stop_telemetry_sampler
trap - EXIT

echo "END_TS=$(now_iso8601)" >> "${meta_file}"
summarize_telemetry
summarize_csv
capture_runtime_tags
merge_runtime_tags_into_summary
echo "Artifacts: ${run_dir}"
