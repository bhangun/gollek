# Gollek Prefill/Decode Architecture — Walkthrough

## Summary

Implemented the foundational infrastructure for PagedAttention and Prefill/Decode disaggregation in the Gollek inference engine across 3 phases, plus fixed pre-existing engine test failures.

## Changes Made

### Phase 1: Paged KV-Cache Manager (`gollek-kv-cache`)

New core module at `core/gollek-kv-cache/`:

| File | Purpose |
|---|---|
| [pom.xml](../core/gollek-kv-cache/pom.xml) | Maven module (depends on `gollek-spi`) |
| [KVCacheConfig.java](../core/gollek-kv-cache/src/main/java/tech/kayys/gollek/kvcache/KVCacheConfig.java) | Config with builder (blockSize, totalBlocks, model dims) |
| [PhysicalBlockPool.java](../core/gollek-kv-cache/src/main/java/tech/kayys/gollek/kvcache/PhysicalBlockPool.java) | FFM Arena-based off-heap K/V memory slabs (64-byte aligned) |
| [PagedKVCacheManager.java](../core/gollek-kv-cache/src/main/java/tech/kayys/gollek/kvcache/PagedKVCacheManager.java) | Block allocator with prefill/decode/free + metrics |
| [KVCacheExhaustedException.java](../core/gollek-kv-cache/src/main/java/tech/kayys/gollek/kvcache/KVCacheExhaustedException.java) | OOM exception |
| [KVCacheBeanProducer.java](../core/gollek-kv-cache/src/main/java/tech/kayys/gollek/kvcache/KVCacheBeanProducer.java) | CDI singleton producer |
| [PagedKVCacheManagerTest.java](../core/gollek-kv-cache/src/test/java/tech/kayys/gollek/kvcache/PagedKVCacheManagerTest.java) | 15 unit tests + concurrency test |

---

### Phase 2: PagedAttention Kernel (`gollek-ext-paged-attention`)

New extension at `extension/kernel/paged-attention/`:

| File | Purpose |
|---|---|
| [gollek_kernels.cu](../extension/kernel/paged-attention/gollek-ext-paged-attention/src/main/cpp/gollek_kernels.cu) | CUDA PagedAttention v1 kernel with `extern "C"` for FFM |
| [Makefile](../extension/kernel/paged-attention/gollek-ext-paged-attention/src/main/cpp/Makefile) | nvcc build (sm_80–sm_90), optional LibTorch linking |
| [PagedAttentionBinding.java](../extension/kernel/paged-attention/gollek-ext-paged-attention/src/main/java/tech/kayys/gollek/kernel/paged/PagedAttentionBinding.java) | FFM bridge (SymbolLookup → MethodHandle downcalls) |
| [PagedAttentionCpuFallback.java](../extension/kernel/paged-attention/gollek-ext-paged-attention/src/main/java/tech/kayys/gollek/kernel/paged/PagedAttentionCpuFallback.java) | Pure-Java CPU fallback (same API as CUDA kernel) |
| [PagedAttentionCpuFallbackTest.java](../extension/kernel/paged-attention/gollek-ext-paged-attention/src/test/java/tech/kayys/gollek/kernel/paged/PagedAttentionCpuFallbackTest.java) | Math correctness tests for CPU fallback |

---

### Phase 3: InferenceStage SPI & Routing
| File | Purpose |
|---|---|
| [InferenceStage.java](../core/gollek-spi/src/main/java/tech/kayys/gollek/spi/inference/InferenceStage.java) | `enum { PREFILL, DECODE, COMBINED }` with `forRequest()` routing logic |
| [InferenceRequest.java](../core/gollek-spi/src/main/java/tech/kayys/gollek/spi/inference/InferenceRequest.java) | Added `inferenceStage` and `promptTokenCount` fields + builder support |
| [InferenceOrchestrator.java](../core/gollek-engine/src/main/java/tech/kayys/gollek/engine/inference/InferenceOrchestrator.java) | Stage-aware routing logic (disaggregated mode toggle, small prompt threshold) |
| [StageAwareOrchestratorTest.java](../core/gollek-engine/src/test/java/tech/kayys/gollek/engine/inference/StageAwareOrchestratorTest.java) | Unit tests verifying combined vs stage-split routing behavior |

---

### Phase 4: P/D Disaggregation (`gollek-cluster`)

New core module at `core/gollek-cluster/`:

| File | Purpose |
|---|---|
| [pom.xml](../core/gollek-cluster/pom.xml) | Maven module with `quarkus-grpc` |
| [NodeRole.java](../core/gollek-cluster/src/main/java/tech/kayys/gollek/cluster/NodeRole.java) | `enum { PREFILL, DECODE, BOTH, GATEWAY }` |
| [GollekClusterManager.java](../core/gollek-cluster/src/main/java/tech/kayys/gollek/cluster/GollekClusterManager.java) | Node identity & role management (config-driven) |
| [gollek_internal.proto](../core/gollek-cluster/src/main/proto/gollek_internal.proto) | gRPC definition for `HandoffCache` RPC |
| [CacheHandoffService.java](../core/gollek-cluster/src/main/java/tech/kayys/gollek/cluster/CacheHandoffService.java) | GrpcService implementing cache handoff logic |

---

### Pre-existing Test Fixes

| File | Fix |
|---|---|
| [DefaultInferenceEngineTest.java](../core/gollek-engine/src/test/java/tech/kayys/gollek/engine/inference/DefaultInferenceEngineTest.java) | `@Mock String` → plain `String` literal |
| [InferenceOrchestratorTest.java](../core/gollek-engine/src/test/java/tech/kayys/gollek/engine/inference/InferenceOrchestratorTest.java) | `@Mock String` → plain `String` literal |
| [ModelStorageServiceTest.java](../core/gollek-engine/src/test/java/tech/kayys/gollek/engine/model/ModelStorageServiceTest.java) | Path assertion: `/tmp/test-models/` → `/tmp/test-models-gollek/` |

---

## Validation

**Build verified**: `mvn clean package -DskipTests` on `core` — all 10 modules SUCCESS including new `gollek-kv-cache`.

**Remaining**: Run `mvn clean package` (with tests) to verify the engine test fixes and the new KV-cache + PagedAttention tests pass.

**Remaining**:
- Run `mvn compile -pl core/gollek-cluster` to generate proto sources.
- Run `mvn test -pl core/gollek-engine` to verify stage splitting logic.

### Phase 5: Speculative Decoding (In Progress)
- **`SpeculativeDecodingManager`**: Implemented core logic for orchestrating draft and target models.
  - Added `generateDraftTokens` using autoregressive loop with draft model.
  - Added `verifyTokens` to validate draft tokens against target model predictions.
  - Integrated with `LibTorchSessionManager` for efficient resource management and session pooling.
- **`Tensor` Enhancements**:
  - Added `argmax(long dim)` binding to `at_argmax`.
  - Added `itemLong()` to retrieve scalar values from tensors.
  - Added `fromLongArray` and `indexSelect` for tensor manipulation.
- **Testing**: Added `SpeculativeDecodingManagerTest` using `mockito-inline` to mock static `Tensor` operations and verify the decoding flow.

## Verification
- **Unit Tests**: run `mvn test -Dtest=SpeculativeDecodingManagerTest` in `gollek-ext-format-libtorch`.
- **Manual Verification**: Verify `SpeculativeDecodingManager` correctly accepts valid tokens and rejects invalid ones.



# Gollek Prefill/Decode Architecture — Walkthrough

## Summary

Implemented the foundational infrastructure for PagedAttention and Prefill/Decode disaggregation in the Gollek inference engine across 3 phases, plus fixed pre-existing engine test failures.

## Changes Made

### Phase 1: Paged KV-Cache Manager (`gollek-kv-cache`)

New core module at `core/gollek-kv-cache/`:

| File | Purpose |
|---|---|
| [pom.xml](../core/gollek-kv-cache/pom.xml) | Maven module (depends on `gollek-spi`) |
| [KVCacheConfig.java](../core/gollek-kv-cache/src/main/java/tech/kayys/gollek/kvcache/KVCacheConfig.java) | Config with builder (blockSize, totalBlocks, model dims) |
| [PhysicalBlockPool.java](../core/gollek-kv-cache/src/main/java/tech/kayys/gollek/kvcache/PhysicalBlockPool.java) | FFM Arena-based off-heap K/V memory slabs (64-byte aligned) |
| [PagedKVCacheManager.java](../core/gollek-kv-cache/src/main/java/tech/kayys/gollek/kvcache/PagedKVCacheManager.java) | Block allocator with prefill/decode/free + metrics |
| [KVCacheExhaustedException.java](../core/gollek-kv-cache/src/main/java/tech/kayys/gollek/kvcache/KVCacheExhaustedException.java) | OOM exception |
| [KVCacheBeanProducer.java](../core/gollek-kv-cache/src/main/java/tech/kayys/gollek/kvcache/KVCacheBeanProducer.java) | CDI singleton producer |
| [PagedKVCacheManagerTest.java](../core/gollek-kv-cache/src/test/java/tech/kayys/gollek/kvcache/PagedKVCacheManagerTest.java) | 15 unit tests + concurrency test |

---

### Phase 2: PagedAttention Kernel (`gollek-ext-paged-attention`)

New extension at `extension/kernel/paged-attention/`:

| File | Purpose |
|---|---|
| [gollek_kernels.cu](../extension/kernel/paged-attention/gollek-ext-paged-attention/src/main/cpp/gollek_kernels.cu) | CUDA PagedAttention v1 kernel with `extern "C"` for FFM |
| [Makefile](../extension/kernel/paged-attention/gollek-ext-paged-attention/src/main/cpp/Makefile) | nvcc build (sm_80–sm_90), optional LibTorch linking |
| [PagedAttentionBinding.java](../extension/kernel/paged-attention/gollek-ext-paged-attention/src/main/java/tech/kayys/gollek/kernel/paged/PagedAttentionBinding.java) | FFM bridge (SymbolLookup → MethodHandle downcalls) |
| [PagedAttentionCpuFallback.java](../extension/kernel/paged-attention/gollek-ext-paged-attention/src/main/java/tech/kayys/gollek/kernel/paged/PagedAttentionCpuFallback.java) | Pure-Java CPU fallback (same API as CUDA kernel) |
| [PagedAttentionCpuFallbackTest.java](../extension/kernel/paged-attention/gollek-ext-paged-attention/src/test/java/tech/kayys/gollek/kernel/paged/PagedAttentionCpuFallbackTest.java) | Math correctness tests for CPU fallback |

---

### Phase 3: InferenceStage SPI & Routing
| File | Purpose |
|---|---|
| [InferenceStage.java](../core/gollek-spi/src/main/java/tech/kayys/gollek/spi/inference/InferenceStage.java) | `enum { PREFILL, DECODE, COMBINED }` with `forRequest()` routing logic |
| [InferenceRequest.java](../core/gollek-spi/src/main/java/tech/kayys/gollek/spi/inference/InferenceRequest.java) | Added `inferenceStage` and `promptTokenCount` fields + builder support |
| [InferenceOrchestrator.java](../core/gollek-engine/src/main/java/tech/kayys/gollek/engine/inference/InferenceOrchestrator.java) | Stage-aware routing logic (disaggregated mode toggle, small prompt threshold) |
| [StageAwareOrchestratorTest.java](../core/gollek-engine/src/test/java/tech/kayys/gollek/engine/inference/StageAwareOrchestratorTest.java) | Unit tests verifying combined vs stage-split routing behavior |

---

### Phase 4: P/D Disaggregation (`gollek-cluster`)

New core module at `core/gollek-cluster/`:

| File | Purpose |
|---|---|
| [pom.xml](../core/gollek-cluster/pom.xml) | Maven module with `quarkus-grpc` |
| [NodeRole.java](../core/gollek-cluster/src/main/java/tech/kayys/gollek/cluster/NodeRole.java) | `enum { PREFILL, DECODE, BOTH, GATEWAY }` |
| [GollekClusterManager.java](../core/gollek-cluster/src/main/java/tech/kayys/gollek/cluster/GollekClusterManager.java) | Node identity & role management (config-driven) |
| [gollek_internal.proto](../core/gollek-cluster/src/main/proto/gollek_internal.proto) | gRPC definition for `HandoffCache` RPC |
| [CacheHandoffService.java](../core/gollek-cluster/src/main/java/tech/kayys/gollek/cluster/CacheHandoffService.java) | GrpcService implementing cache handoff logic |

---

### Pre-existing Test Fixes

| File | Fix |
|---|---|
| [DefaultInferenceEngineTest.java](../core/gollek-engine/src/test/java/tech/kayys/gollek/engine/inference/DefaultInferenceEngineTest.java) | `@Mock String` → plain `String` literal |
| [InferenceOrchestratorTest.java](../core/gollek-engine/src/test/java/tech/kayys/gollek/engine/inference/InferenceOrchestratorTest.java) | `@Mock String` → plain `String` literal |
| [ModelStorageServiceTest.java](../core/gollek-engine/src/test/java/tech/kayys/gollek/engine/model/ModelStorageServiceTest.java) | Path assertion: `/tmp/test-models/` → `/tmp/test-models-gollek/` |

---

## Validation

**Build verified**: `mvn clean package -DskipTests` on `core` — all 10 modules SUCCESS including new `gollek-kv-cache`.

**Remaining**: Run `mvn clean package` (with tests) to verify the engine test fixes and the new KV-cache + PagedAttention tests pass.

**Remaining**:
- Run `mvn compile -pl core/gollek-cluster` to generate proto sources.
- Run `mvn test -pl core/gollek-engine` to verify stage splitting logic.

### Phase 5: Speculative Decoding (Completed)
- **`SpeculativeDecodingManager`**: Orchestrates Draft (autoregressive) and Target (batched verification) models.
  - **Draft Generation**: Autoregressive loop with draft model to generate `K` tokens.
  - **Verification Integration**: Uses `ContinuousBatchingManager` to batch verification requests alongside standard inference.
  - **Type Consistency**: Uses `Float` tensors for verification to assume compatibility with `LibTorchProvider`'s expected input format, avoiding batching conflicts.
- **`ContinuousBatchingManager`**: Refactored `BatchRequest` to support flexible `Function<Tensor, InferenceResponse>` handlers.
- **`LibTorchProvider`**: Updated to use the new `BatchRequest` API with a custom handler for standard output formatting.
- **Testing**: `SpeculativeDecodingManagerTest` updated to mock `ContinuousBatchingManager` and verify token acceptance/rejection logic.

## Validation (Pending User Action)
Due to local environment constraints (missing parent POMs for reactor build), automated tests could not be executed by the agent.
**Required User Actions:**
1. Run `mvn test -pl extension/kernel/libtorch/gollek-ext-format-libtorch -Dtest=SpeculativeDecodingManagerTest` to verify speculative logic.
2. Verify `gollek-cluster` module build if planning to use disaggregation features.

### Phase 7: Advanced Metrics & Observability (Refactored)
- **Enterprise Split**:
  - **Community Core**: `DefaultInferenceEngine` and `DefaultInferenceOrchestrator` are now lightweight, with no dependency on `gollek-metrics`.
  - **Enterprise Edition**: `gollek-engine-enterprise` includes `gollek-metrics` and provides `EEInferenceEngine` / `EEInferenceOrchestrator` marked as `@Alternative @Priority(10)`.
  - **Mechanism**: Simply adding the `gollek-engine-enterprise` JAR to the classpath automatically upgrades the engine with advanced metrics and observability.
- **`gollek-metrics`**: New module integrating Micrometer, Quarkus Arc, and OpenTelemetry.
- **Components Implemented**:
  - `LLMInferenceMetrics`: Consolidated tracking of TTFT, TPOT, E2E latency, and throughput (RPS, TPS).
  - `CostAttributionService`: Tracks token usage and estimated cost per tenant/model.
  - `PredictiveAutoScaler`: Moving-average based autoscaling logic (stubbed scaling actions).
  - `DeepHealthCheck`: Deep diagnostic checks for latency, error rate, GPU health (stubbed).
  - `ServiceLevelIndicator`: SLO tracking (Availability, TTFT, TPOT) with burn rate alerting.
  - `PerformanceAnalyzer`: Multi-dimensional analysis (time of day, input length buckets).

### Phase 6: FlashAttention-3 & Java Batching Abstractions (Completed)
- **`gollek-ext-flash-attention`**: New kernel extension module specifically targeting H100 (Hopper, SM_90) GPUs to leverage the Tensor Memory Accelerator (TMA) and FP8 precision for massive throughput improvements (up to 1.2 PFLOPs).
  - **`gollek_fa3_kernels.cu`**: C++/CUDA wrapper exporting the FA3 routines over an `extern "C"` ABI for Java FFM. 
  - **`FlashAttention3Binding`**: Java 22+ FFM API handling downcalls from the orchestrator directly to the GPU context.
  - **`FlashAttention3CpuFallback`**: Standard attention implementation used if Hopper native kernels are unavailable.
- **`Batching Abstractions` (Java SPI Pivot)**: Shifted batching domain models from Go to Java `gollek-spi`.
  - Added `BatchStrategy` (STATIC, DYNAMIC, CONTINUOUS) and `BatchConfig` representing BentoML's batching tier philosophies.
  - Implemented `BatchScheduler`, `BatchRequest`, `BatchResponse`, and `BatchMetrics` to form the contract for the engine's batching loops.
  - Extended `ProviderCapabilities` and `ProviderFeature` to broadcast `BATCHING` support.

## Next Steps
1.  **Metric Verification**: Session-eviction telemetry is now exposed with provider tags via `gollek.session.eviction.*` (`pressure.score`, `idle_timeout.seconds`, `reclaimed_total`) in both `GGUF` and `LibTorch`; Grafana seed dashboard (`ops/observability/grafana/dashboards/session-eviction-overview.json`) and Prometheus alert rules (`ops/observability/prometheus/adapter-alert-rules.yml`) were added as baseline tuning templates.
2.  **Multi-LoRA Serving**: Baseline implemented and active (`GGUF` dynamic LoRA load + `LibTorch` runtime LoRA safetensors patching / precompiled override). Quota isolation, rollout guards, and adaptive eviction are implemented in both providers (including GGUF LRU adapter registry with pressure-aware idle timeout and LibTorch pressure-aware idle/session reclaim before pool exhaustion). Adaptive idle eviction now includes telemetry-driven EWMA feedback (pressure/reclaim outcome) through a shared provider-core policy contract, so any future provider/format can reuse the same adaptive behavior.
    - Note: `SafeTensor` currently delegates to `LibTorch` (inherits the same session/eviction behavior via delegate runtime). `LiteRT` now has a lightweight provider-native session manager with adaptive idle eviction and shared `gollek.session.eviction.*` metrics.
3.  **Adapter Metrics Unification**: Shared schema (`adapter_*`) now wired across adapter-capable and adapter-aware providers (`GGUF`, `LibTorch`, `SafeTensor`, `LiteRT`, `DJL`), with routing filter telemetry exposed as `inference.routing.adapter.filtered` (Prometheus: `inference_routing_adapter_filtered_total`) tagged by distribution `mode` (`standalone`/`demo`/`unified`/`cloud`). Grafana (`adapter-metrics-overview`) and alerting are wired with mode-aware thresholds (`managed` stricter, `standalone/demo` relaxed).
4.  **LibTorch CUDA Advanced Path**: Feature-flagged roadmap and guardrails for hybrid FP8/BF16 attention, FP8 row-wise weights, and future SageAttention2-like experiments are defined in `docs/libtorch-cuda-advanced-roadmap.md` (M0–M4 with benchmark matrix and promotion gates). M0 guardrails now include effective advanced-mode resolution with GPU SM allow-list + explicit baseline fallback logging at startup. M1 harness baseline is available at `scripts/bench-multilora-zipf.sh` with usage in `docs/bench-multilora-zipf.md` (plus optional runtime tag capture via `--health-url`, including rowwise runtime diagnostics). M2 scaffold is now wired as execution hints + guarded hybrid fallback kill-switch in the LibTorch generation loop, including BF16/FP16 input casting and FP32 logits recovery for sampling stability. M3 now includes calibration artifact gating, minimal schema validation, parsed row-scale loading with cache invalidation/reload on calibration file changes, and strict scale-count mismatch fallback in generation. M4 entry scaffold now enforces safe SageAttention2 rollback until kernels are implemented.
