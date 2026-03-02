# Gollek LibTorch CUDA Advanced Roadmap

## Objective

Incrementally close the performance gap for Multi-LoRA serving on Hopper/Ada GPUs while keeping `gguf` and baseline `libtorch` stable.

This roadmap tracks the feature-flagged `libtorch-cuda-advanced` path for:
- Hybrid FP8/BF16 attention
- FP8 row-wise quantization
- SageAttention2-like experimental path (later milestone)

## Safety Principles

1. Baseline remains default:
   - `gguf` path unchanged.
   - `libtorch` baseline path unchanged.
2. Advanced path is opt-in only (`advanced.enabled=false` by default).
3. Unsupported GPU/driver must auto-fallback to baseline.
4. Promotion to default behavior is not allowed in this roadmap.

## Feature Flags

Configuration prefix: `libtorch.provider.advanced`

- `enabled` (`false`): master switch for advanced path.
- `attention-mode` (`baseline`): `baseline|hybrid_fp8_bf16`.
- `fp8-rowwise-enabled` (`false`): row-wise quantized weight path.
- `fp8-rowwise-allowed-tenants` (empty): optional rowwise canary tenant allow-list.
- `fp8-rowwise-allowed-models` (empty): optional rowwise canary model allow-list.
- `fp8-rowwise-blocked-tenants` (empty): optional rowwise canary tenant deny-list (takes precedence).
- `fp8-rowwise-blocked-models` (empty): optional rowwise canary model deny-list (takes precedence).
- `sage-attention2-enabled` (`false`): experimental SA2-like path.
- `sage-attention2-allowed-tenants` (empty): optional canary tenant allow-list.
- `sage-attention2-allowed-models` (empty): optional canary model allow-list.
- `sage-attention2-blocked-tenants` (empty): optional canary tenant deny-list (takes precedence).
- `sage-attention2-blocked-models` (empty): optional canary model deny-list (takes precedence).
- `allowed-gpu-sm` (`89,90`): comma-separated GPU SM allow-list.

Runtime env mappings are available in:
- `runtime/gollek-runtime-cloud/src/main/resources/application.yaml`
- `runtime/gollek-runtime-unified/src/main/resources/application.properties`
- `runtime/gollek-runtime-standalone/src/main/resources/application.properties`

## Milestones

## M0: Safety Envelope (Week 1)

Deliverables:
1. Flag surface + config validation (done as baseline in this phase).
2. GPU capability guard:
   - Parse `allowed-gpu-sm`.
   - If GPU SM is unsupported, force baseline path.
3. Structured logging:
   - Emit effective advanced mode at startup and on model/session init.

Acceptance:
1. No behavior change when all advanced flags are default.
2. Explicit fallback logs on unsupported GPU.
3. Existing libtorch and gguf tests remain green.

## M1: Benchmark Harness (Weeks 1-2)

Goal:
Produce reliable baselines before kernel changes.

Deliverables:
1. `scripts/bench-multilora-zipf.sh` harness (provider-agnostic flags) with:
   - Zipf adapter popularity (`alpha`: 0.8, 1.0, 1.2).
   - Active adapter cardinality (32, 128, 512).
   - Workload mixes:
     - Prefill-heavy: 70/30 prefill/decode
     - Balanced: 50/50
     - Decode-heavy: 30/70
   - Advanced profile segmentation:
     - `baseline`
     - `hybrid-fp8-bf16`
     - `sageattention2-intent` (explicit intent tracking for rollback-only SA2 stage)
2. Outputs:
   - TTFT p50/p95
   - TPOT p50/p95
   - Throughput (tokens/s, req/s)
   - Adapter switch latency
   - Error rate
   - Host/GPU telemetry rollups (CPU load, memory usage, GPU utilization/memory when available)
   - Keep memory bandwidth in external GPU telemetry (Nsight/DCGM)
3. Reproducible config snapshots (`json`/`yaml`) per run.
4. Baseline-vs-candidate comparison report via `scripts/bench-compare.sh` (text + JSON deltas).
5. Matrix gate runner via `scripts/bench-matrix-advanced.sh` (baseline + hybrid + SA2-intent + promotion gates).

Acceptance:
1. Two consecutive runs differ by <= 5% for primary KPIs.
2. Results persisted in `ops/benchmarks/` with run metadata.

## M2: Hybrid FP8/BF16 Attention (Weeks 3-5)

Goal:
First real optimization path with constrained risk.

Deliverables:
1. `attention-mode=hybrid_fp8_bf16` implementation.
2. Conservative precision split:
   - Sensitive reductions/accumulators remain BF16.
   - Selected matmul/attention subpath in FP8.
3. Runtime kill-switches:
   - Immediate fallback on numerical instability signals.
4. Benchmark comparisons against M1 baseline.

Acceptance:
1. >= 20% throughput gain on at least one Hopper profile.
2. Quality drift within agreed threshold (task-specific metric, default < 0.5%).
3. No increase in error rate under 24h soak test.

## M3: FP8 Row-wise Weights (Weeks 6-7)

Goal:
Reduce bandwidth pressure and improve kernel efficiency.

Deliverables:
1. Row-wise scale/cast path behind `fp8-rowwise-enabled`.
2. Calibration artifact format and lifecycle (load/cache/invalidate).
3. Backward-compatible fallback when calibration artifacts are missing.

Acceptance:
1. Additional gain over M2 for bandwidth-bound profiles.
2. No unacceptable quality drift vs M2 baseline.
3. Stable under adapter churn and mixed-tenant load.

## M4: SageAttention2-like Experimental Path (Backlog)

Goal:
Explore INT8/INT4-like attention path only after M2/M3 stability.

Deliverables:
1. Experimental implementation behind `sage-attention2-enabled`.
2. Narrow allow-list (GPU + model families) for early access.
3. Explicit “experimental” operational profile and rollback guidance.

Acceptance:
1. Separate benchmark and quality report from baseline/M2/M3.
2. Must pass soak and correctness gates before widening rollout.

## Test Matrix

Mandatory matrix per milestone:
1. GPU classes:
   - A10/Ada class
   - A100
   - H100
2. Modes:
   - Baseline (`advanced.enabled=false`)
   - Advanced mode under test
3. Workloads:
   - Prefill-heavy / balanced / decode-heavy
   - Low and high adapter cardinality
4. Reliability:
   - Cold start
   - Adapter churn bursts
   - 24h soak

## Promotion Gates

A milestone is complete only if:
1. KPI gain meets target.
2. Quality drift within threshold.
3. Error/timeout regressions are not observed.
4. Rollback to baseline is verified in runtime tests.

## Current Status

- M0: In progress (flags/config added, defaults safe, GPU SM allow-list guard + explicit startup fallback logging implemented in LibTorch provider).
- M1: In progress (`scripts/bench-multilora-zipf.sh` and `docs/bench-multilora-zipf.md` added).
- M2: In progress (hybrid execution hint plumbing + guarded fallback kill-switch scaffold in LibTorch generation path with BF16/FP16 bridge + FP32 logits recovery).
- M3: In progress (rowwise calibration schema validation, parsed scale loading, cache lifecycle with file-change reload/invalidate, and strict runtime mismatch fallback are implemented).
- M4: Entry scaffold in progress (flag remains opt-in, resolver now forces SageAttention2 rollback until kernels exist).

Current M2 note:
- Hybrid path now performs real precision bridging around forward pass: input cast to BF16 (or FP16 fallback), then logits cast back to FP32 for stable sampling. Any runtime failure triggers immediate fallback to baseline via kill-switch.
- Benchmark harness can optionally capture runtime advanced-mode tags via `--health-url` (`runtime-tags.json`) to correlate run artifacts with effective mode.
- Benchmark harness now also captures optional host/GPU telemetry (`--telemetry`, `--gpu-telemetry`) into `telemetry.csv` and merges summary rollups into `summary.txt` / `summary.json`.
- Benchmark harness now supports `--advanced-profile` so baseline/hybrid/SA2-intent runs are explicitly tagged in payloads and artifacts (`advanced_profile`, expected-mode fields).
- Matrix runner now available: `scripts/bench-matrix-advanced.sh` executes the 3-profile matrix and emits gate pass/fail artifacts for CI.
- Matrix runner now supports runtime-tag strictness control (`--runtime-tag-gate auto|on|off`) so SA2-intent requested/effective validation can be required when health tags are available.
- Operations guidance is documented in `docs/benchmark-operations-runbook.md` (workflow policy, thresholds, and gate triage).

Current M3 note (entry):
- FP8 rowwise activation is now gated by calibration artifact presence (`<model>.fp8.json` / `<model>.fp8calib.json` / `<model>.calibration.json`). Missing calibration forces rowwise off with explicit reason while keeping hybrid/baseline safe.
- Calibration validation now enforces minimal schema (`version` + non-empty numeric `row_scales`/`scales`) before rowwise mode can activate.
- Calibration lifecycle now supports load + cache + invalidate semantics with deterministic reload when calibration file content changes.
- Runtime strictness: hybrid path checks calibration scale count against logits vocab size; mismatch triggers immediate guarded fallback to baseline path.
- Request-time canary controls now apply to rowwise path with explicit reasons:
  - allow-list misses: `fp8.rowwise.canary.blocked.tenant|model`
  - deny-list hits (higher precedence): `fp8.rowwise.canary.denied.tenant|model`

Current M0 note:
- Effective advanced-mode resolution currently supports explicit SM override via `-Dgollek.libtorch.gpu.sm=<sm>` or `GOLLEK_LIBTORCH_GPU_SM=<sm>` and native symbol probing when available. If SM cannot be resolved, runtime safely falls back to baseline.

Current M4 note (entry scaffold):
- `sage-attention2-enabled` is now treated as an experimental intent flag only; resolver applies strict rollback (forced disable) until SageAttention2 kernels are implemented.
- If SageAttention2 is requested alone, provider falls back to baseline (`reason=sageattention2.rollback`).
- If SageAttention2 is requested together with another implemented advanced mode (for example `hybrid_fp8_bf16`), resolver keeps the implemented mode active and rolls back only SageAttention2.
- Canary control is now available via request-time allow-lists (`sage-attention2-allowed-tenants`, `sage-attention2-allowed-models`), producing explicit rollback reasons (`sageattention2.canary.blocked.tenant|model`) when outside scope.
- Canary deny-lists (`sage-attention2-blocked-tenants`, `sage-attention2-blocked-models`) now take precedence over allow-lists, with explicit reasons (`sageattention2.canary.denied.tenant|model`).
- Runtime observability now exposes SageAttention2 requested/active/reason tags in health and benchmark runtime tags (`advanced_sage_attention2_*`) so rollout intent and rollback cause are explicit.
