# Benchmark Operations Runbook

This runbook defines how to operate Gollek benchmark matrix workflows in development and release pipelines.

## Workflow Modes

1. Smoke (`bench-matrix-smoke.yml`)
- Purpose: deterministic CI validation of benchmark scripts and matrix orchestration.
- Trigger: PR/push/manual.
- Backend: mock local inference + health server.
- Gate mode: strict runtime-tag gate (`runtime-tag-gate=on` in CI smoke).
- Expected use: merge safety for tooling changes.

2. Strict (`bench-matrix-strict.yml`)
- Purpose: real endpoint validation with production-like thresholds.
- Trigger: manual (`workflow_dispatch`).
- Backend: actual runtime endpoint + health endpoint.
- Gate mode: usually `runtime-tag-gate=on`.
- Expected use: pre-release and periodic performance checks.
- Trend output: strict workflow now auto-generates trend snapshot files under the run artifact (`trends/matrix-gates.csv`, `trends/matrix-gates.json`).
- Alert output: strict workflow now generates gate alert files (`gate-alert.txt`, `gate-alert.md`) and can send webhook notifications on failure.

## Threshold Presets

Use these as starting points and tune by model family.

1. Dev preset (fast iteration)
- `requests=100`
- `concurrency=8`
- `adapters=64`
- `hybrid_min_throughput_pct=10`
- `hybrid_max_latency_regress_pct=20`
- `hybrid_max_error_regress_abs=0.01`

2. Staging preset (balanced confidence)
- `requests=500`
- `concurrency=32`
- `adapters=128`
- `hybrid_min_throughput_pct=20`
- `hybrid_max_latency_regress_pct=10`
- `hybrid_max_error_regress_abs=0.005`

3. Production gate preset (strict)
- `requests=1000`
- `concurrency=64`
- `adapters=256`
- `hybrid_min_throughput_pct=20`
- `hybrid_max_latency_regress_pct=8`
- `hybrid_max_error_regress_abs=0.003`

## Failure Playbook

When `matrix_result=FAIL`:

1. Check gate file first
- Open `gate-alert.txt` first, then `gate-summary.txt` and `gate-summary.json`.
- Identify which gate failed:
  - `hybrid_throughput`
  - `hybrid_latency`
  - `hybrid_error`
  - `sa2_runtime_required_pass`

2. If throughput gate fails
- Validate workload parity (same `requests`, `concurrency`, `adapters`, `zipf_alpha`, `mix`).
- Compare health runtime tags to ensure correct advanced mode was active.
- Re-run once to detect transient noise.

3. If latency gate fails
- Inspect p95 split in comparison report (`latency_all_p95_ms`, `ttft_p95_ms`, `tpot_p95_ms`).
- Check adapter churn pressure (`adapter_switch_ratio`) and telemetry spikes.
- Reduce concurrency temporarily to isolate saturation effects.

4. If error gate fails
- Inspect `results.csv` error bodies and status codes.
- Confirm calibration artifacts and feature flags for hybrid/rowwise path.
- Roll back to baseline flags while triaging.

5. If SA2 runtime gate fails
- Verify `--health-url` points to the runtime exposing `advanced_*` tags.
- Confirm SA2 intent requests are visible (`advanced_sage_attention2_requested=true`).
- Confirm effective attention mode remains `baseline` until SA2 kernels are implemented.

## Alerting

1. Local parser
- `scripts/bench-gate-alert.sh --gate-json <path>/gate-summary.json --out-md <path>/gate-alert.md`

2. Strict CI webhook (optional)
- Configure secret: `GOLEK_BENCH_ALERT_WEBHOOK`
- On matrix failure, strict workflow posts concise alert text to this webhook URL.

3. Publish latest status for website badge

```bash
./scripts/publish-latest-gate-summary.sh \
  --gate-json /path/to/strict-run/gate-summary.json \
  --output ../website/gollek-ai.github.io/assets/data/latest-gate-summary.json
```

Then commit/publish the website repo so `/assets/data/latest-gate-summary.json` is updated.

## Trend Snapshot

Generate trend snapshots from historical matrix runs:

```bash
./scripts/bench-trend-snapshot.sh \
  --matrix-root ops/benchmarks/matrix \
  --out-csv ops/benchmarks/trends/matrix-gates.csv \
  --out-json ops/benchmarks/trends/matrix-gates.json \
  --limit 100
```

Outputs:
- `matrix-gates.csv`: quick spreadsheet view of recent gates and deltas.
- `matrix-gates.json`: machine-readable summary with averages and pass/fail counts.

## Recommended Team Policy

1. Required for PR merge
- `bench-matrix-smoke.yml`

2. Required before release cut
- one successful `bench-matrix-strict.yml` run with staging preset or stricter

3. Monitoring cadence
- run strict matrix weekly for active model families
- refresh trend snapshot after each strict run

Note: with current strict CI wiring, trend snapshot refresh is automatic per run. Manual `bench-trend-snapshot.sh` is still useful for local/offline aggregation.
