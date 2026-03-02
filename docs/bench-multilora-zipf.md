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
- `--advanced-profile`: benchmark profile label with intent semantics:
  - `baseline`
  - `hybrid-fp8-bf16`
  - `sageattention2-intent` (tracks SA2 rollout intent while runtime may rollback)
- `--telemetry`: host telemetry mode (`auto|on|off`, default `auto`).
- `--gpu-telemetry`: GPU telemetry mode (`auto|on|off`, default `auto`, uses `nvidia-smi` when present).
- `--sample-interval-sec`: telemetry sample interval (default `1`).

These flags let the same harness run against different provider/API conventions without changing source code.

## KPI Outputs

Each run stores:
- `config.json`: reproducible input snapshot.
- `plan.txt`: generated request plan (Zipf + phase assignment).
- `results.csv`: per-request status and latency.
- `summary.txt` and `summary.json`:
  - profile metadata:
    - `advanced_profile`
    - `profile_expected_mode`
    - `profile_expected_sage_requested`
    - `profile_expected_sage_active`
  - total/ok/fail/error_rate
  - latency p50/p95
  - TTFT p50/p95 (prefill phase)
  - TPOT p50/p95 (decode phase)
  - throughput req/s
  - throughput tokens/s (estimated from configured token budgets)
  - adapter switch count and ratio (from generated plan)
  - host telemetry rollups:
    - `host_load_1_avg`
    - `host_mem_used_mb_avg`
    - `host_mem_used_mb_peak`
  - GPU telemetry rollups (when `nvidia-smi` is available):
    - `gpu_util_avg`
    - `gpu_util_p95`
    - `gpu_mem_used_mb_peak`
- `telemetry.csv`:
  - timestamped host/GPU samples captured during the run
- `telemetry-summary.txt`:
  - reduced telemetry metrics merged into `summary.*`
- `results.csv`:
  - includes `benchmark_profile` column for per-request segmentation in downstream analytics
- `runtime-tags.json` (when `--health-url` is set):
  - `advanced_effective_enabled`
  - `advanced_attention_mode`
  - `advanced_reason`
  - `advanced_detected_gpu_sm`
  - `advanced_sage_attention2_requested`
  - `advanced_sage_attention2_active`
  - `advanced_sage_attention2_reason`
  - `advanced_fp8_rowwise_requested`
  - `advanced_fp8_rowwise_active`
  - `advanced_fp8_rowwise_reason`
  - `advanced_fp8_rowwise_scale_count`
  - `advanced_fp8_rowwise_scale_mean`
  - `advanced_fp8_rowwise_calibration_source`

## Notes

- TTFT and TPOT are approximated from phase split:
  - `prefill` requests map to TTFT latency.
  - `decode` requests map to TPOT latency.
- GPU memory bandwidth is still not collected by this script; capture that with Nsight/DCGM or your GPU telemetry stack when needed.

## Baseline vs Candidate Comparison

Script: `scripts/bench-compare.sh`

Use it to compare one baseline run against one or more candidate runs and compute KPI deltas.

```bash
./scripts/bench-compare.sh \
  --baseline ops/benchmarks/zipf-balanced-a1.0-n500-c32-<baseline-ts> \
  --candidate ops/benchmarks/zipf-balanced-a1.0-n500-c32-<hybrid-ts> \
  --candidate ops/benchmarks/zipf-balanced-a1.0-n500-c32-<sa2-intent-ts>
```

Outputs under `ops/benchmarks/comparisons/<report-name>/`:
- `report.txt`: human-readable table
- `report.json`: structured comparison for automation/CI

## Matrix + Gates

Script: `scripts/bench-matrix-advanced.sh`

Runs all three profiles (`baseline`, `hybrid-fp8-bf16`, `sageattention2-intent`) and evaluates promotion gates.

```bash
./scripts/bench-matrix-advanced.sh \
  --endpoint http://localhost:8080/api/v1/inference \
  --model llama-3-8b \
  --requests 500 \
  --concurrency 32 \
  --adapters 128 \
  --zipf-alpha 1.0 \
  --mix balanced \
  --health-url http://localhost:8080/q/health/ready
```

Runtime-tag gate control:
- `--runtime-tag-gate auto|on|off` (default `auto`)
- `auto`: strict runtime-tag gating is enabled only when `--health-url` is set
- `on`: strict runtime-tag gating always required
- `off`: runtime-tag checks are reported but not required for matrix pass

Default hybrid gates:
- throughput uplift: `>= 20%` (`req/s` OR `tokens/s`)
- latency regression cap: `<= 10%` (`latency_all_p95_ms`)
- error regression cap: `<= 0.005` absolute

Outputs under `ops/benchmarks/matrix/<matrix-name>/`:
- `matrix-summary.json`: run/report index
- `gate-summary.txt`: human-readable gate results
- `gate-summary.json`: machine-readable gate results
- `compare/baseline-vs-candidates/report.{txt,json}`: KPI deltas

Exit code:
- `0` when all required gates pass
- `1` when any required gate fails

## CI Workflows

- `.github/workflows/bench-matrix-smoke.yml`
  - PR/push/manual smoke validation using a local mock server.
  - Verifies scripts, matrix orchestration, strict runtime-tag gate wiring, and artifact generation.
- `.github/workflows/bench-matrix-strict.yml`
  - Manual strict run (`workflow_dispatch`) against a real endpoint.
  - Inputs expose workload size and gate thresholds.
  - Optional auth via `GOLEK_BENCH_API_KEY` secret.
  - Auto-generates trend snapshot files under run artifacts (`trends/matrix-gates.{csv,json}`).
  - Auto-generates gate alert files (`gate-alert.txt`, `gate-alert.md`).
  - Optional failure webhook via `GOLEK_BENCH_ALERT_WEBHOOK` secret.
  - Supports trend lookback input (`trend_limit`, default 100).
  - Uploads run artifacts with configurable retention (`artifact_retention_days`, default 30).

## Operations Runbook

See `docs/benchmark-operations-runbook.md` for:
- smoke vs strict workflow policy
- threshold presets (dev/staging/prod)
- gate failure playbook
- trend snapshot procedure
