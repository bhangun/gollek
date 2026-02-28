# Multi-LoRA Zipf Benchmark Harness

Script: `scripts/bench-multilora-zipf.sh`

Purpose:
- Generate Zipf-distributed adapter traffic for Multi-LoRA scenarios.
- Exercise prefill-heavy, balanced, and decode-heavy mixes.
- Persist reproducible benchmark artifacts under `ops/benchmarks/<run_id>/`.

## Quick Start

```bash
./scripts/bench-multilora-zipf.sh \
  --endpoint http://localhost:8080/api/v1/inference \
  --model llama-3-8b \
  --requests 500 \
  --concurrency 32 \
  --adapters 128 \
  --zipf-alpha 1.0 \
  --mix balanced
```

## Provider-Adaptive Options

- `--adapter-param-key`: override adapter key in payload parameters (default `adapter_id`).
- `--api-key-header`: override auth header key (default `X-API-Key`).
- `--api-key`: auth header value.
- `--health-url`: optional runtime health endpoint; if provided, benchmark captures effective advanced-mode tags (enabled/mode/reason/SM) into run artifacts.

These flags let the same harness run against different provider/API conventions without changing source code.

## KPI Outputs

Each run stores:
- `config.json`: reproducible input snapshot.
- `plan.txt`: generated request plan (Zipf + phase assignment).
- `results.csv`: per-request status and latency.
- `summary.txt` and `summary.json`:
  - total/ok/fail/error_rate
  - latency p50/p95
  - TTFT p50/p95 (prefill phase)
  - TPOT p50/p95 (decode phase)
  - throughput req/s
  - throughput tokens/s (estimated from configured token budgets)
  - adapter switch count and ratio (from generated plan)
- `runtime-tags.json` (when `--health-url` is set):
  - `advanced_effective_enabled`
  - `advanced_attention_mode`
  - `advanced_reason`
  - `advanced_detected_gpu_sm`
  - `advanced_sage_attention2_requested`
  - `advanced_sage_attention2_active`
  - `advanced_sage_attention2_reason`
  - `advanced_fp8_rowwise_active`
  - `advanced_fp8_rowwise_reason`
  - `advanced_fp8_rowwise_scale_count`
  - `advanced_fp8_rowwise_scale_mean`
  - `advanced_fp8_rowwise_calibration_source`

## Notes

- TTFT and TPOT are approximated from phase split:
  - `prefill` requests map to TTFT latency.
  - `decode` requests map to TPOT latency.
- GPU utilization and memory bandwidth are not collected by this script; capture those with your GPU telemetry stack during the same run.
