Yes, that is still the right plan. I’d run it in this order with hard gates.

**M0: Safety Envelope (1 week)**
1. Keep `gguf` default unchanged.
2. Add opt-in flag namespace:
   - `gollek.libtorch.advanced.enabled=false`
   - `gollek.libtorch.advanced.attention.mode=baseline|hybrid_fp8_bf16`
   - `gollek.libtorch.advanced.fp8.rowwise.enabled=false`
   - `gollek.libtorch.advanced.sage2.enabled=false`
   - `gollek.libtorch.advanced.allowed-gpu-sm=89,90`
3. Runtime guard: if unsupported GPU/driver, auto-fallback to baseline.

**M1: Benchmark Harness First (1-2 weeks)**
1. Build Multi-LoRA Zipf harness (prefill-heavy + decode-heavy mixes).
2. Workload params:
   - Zipf alpha: `0.8, 1.0, 1.2`
   - Active adapters: `32, 128, 512`
   - Mix: prefill-heavy (70/30), balanced (50/50), decode-heavy (30/70)
3. Output KPIs:
   - TTFT p50/p95, TPOT p50/p95
   - tokens/s, requests/s
   - adapter switch latency
   - GPU util + memory BW
   - error rate / numeric drift

**M2: Hybrid FP8/BF16 Attention (2-3 weeks)**
1. Implement only one optimization path first: `hybrid_fp8_bf16`.
2. Keep numerically sensitive ops in BF16; FP8 only where stable.
3. Acceptance gate:
   - `>=20%` throughput gain on H100/Ada test set
   - quality drift within tolerance (`<0.5%` target metric delta)
   - no regression on non-advanced path.

**M3: FP8 Rowwise Weights (2 weeks)**
1. Add rowwise scaling + calibration flow.
2. Persist calibration artifacts per model/adapterset.
3. Gate:
   - memory reduction + throughput gain without new instability.

**M4: SageAttention2-like Path (later, 3-5 weeks)**
1. Start experimental only behind `sage2.enabled`.
2. Limited model/device matrix first.
3. Promote only after long soak + quality parity.

**Test Matrix (required each milestone)**
1. GPU: A10/A100/H100 + Ada consumer.
2. Providers: `libtorch` advanced on/off, `gguf` baseline.
3. Adapters: hot/cold, churn bursts, tenant isolation.
4. Soak: 24h stability with mixed Zipf traffic.

This keeps GGUF stable, gives measurable progress early, and avoids jumping too quickly to SageAttention2 complexity.