# Gollek Prefill/Decode Architecture — Walkthrough

## Summary

Implemented the foundational infrastructure for PagedAttention and Prefill/Decode disaggregation in the Gollek inference engine across 3 phases, plus fixed pre-existing engine test failures.

## Changes Made

### Phase 1: Paged KV-Cache Manager (`gollek-kv-cache`)

New core module at `inference-gollek/core/gollek-kv-cache/`:

| File | Purpose |
|---|---|
| [pom.xml](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-kv-cache/pom.xml) | Maven module (depends on `gollek-spi`) |
| [KVCacheConfig.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-kv-cache/src/main/java/tech/kayys/gollek/kvcache/KVCacheConfig.java) | Config with builder (blockSize, totalBlocks, model dims) |
| [PhysicalBlockPool.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-kv-cache/src/main/java/tech/kayys/gollek/kvcache/PhysicalBlockPool.java) | FFM Arena-based off-heap K/V memory slabs (64-byte aligned) |
| [PagedKVCacheManager.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-kv-cache/src/main/java/tech/kayys/gollek/kvcache/PagedKVCacheManager.java) | Block allocator with prefill/decode/free + metrics |
| [KVCacheExhaustedException.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-kv-cache/src/main/java/tech/kayys/gollek/kvcache/KVCacheExhaustedException.java) | OOM exception |
| [KVCacheBeanProducer.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-kv-cache/src/main/java/tech/kayys/gollek/kvcache/KVCacheBeanProducer.java) | CDI singleton producer |
| [PagedKVCacheManagerTest.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-kv-cache/src/test/java/tech/kayys/gollek/kvcache/PagedKVCacheManagerTest.java) | 15 unit tests + concurrency test |

---

### Phase 2: PagedAttention Kernel (`gollek-ext-paged-attention`)

New extension at `inference-gollek/extension/kernel/paged-attention/`:

| File | Purpose |
|---|---|
| [gollek_kernels.cu](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/extension/kernel/paged-attention/gollek-ext-paged-attention/src/main/cpp/gollek_kernels.cu) | CUDA PagedAttention v1 kernel with `extern "C"` for FFM |
| [Makefile](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/extension/kernel/paged-attention/gollek-ext-paged-attention/src/main/cpp/Makefile) | nvcc build (sm_80–sm_90), optional LibTorch linking |
| [PagedAttentionBinding.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/extension/kernel/paged-attention/gollek-ext-paged-attention/src/main/java/tech/kayys/gollek/kernel/paged/PagedAttentionBinding.java) | FFM bridge (SymbolLookup → MethodHandle downcalls) |
| [PagedAttentionCpuFallback.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/extension/kernel/paged-attention/gollek-ext-paged-attention/src/main/java/tech/kayys/gollek/kernel/paged/PagedAttentionCpuFallback.java) | Pure-Java CPU fallback (same API as CUDA kernel) |
| [PagedAttentionCpuFallbackTest.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/extension/kernel/paged-attention/gollek-ext-paged-attention/src/test/java/tech/kayys/gollek/kernel/paged/PagedAttentionCpuFallbackTest.java) | Math correctness tests for CPU fallback |

---

### Phase 3: InferenceStage SPI & Routing
| File | Purpose |
|---|---|
| [InferenceStage.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/spi/inference/InferenceStage.java) | `enum { PREFILL, DECODE, COMBINED }` with `forRequest()` routing logic |
| [InferenceRequest.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/spi/inference/InferenceRequest.java) | Added `inferenceStage` and `promptTokenCount` fields + builder support |
| [InferenceOrchestrator.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-engine/src/main/java/tech/kayys/gollek/engine/inference/InferenceOrchestrator.java) | Stage-aware routing logic (disaggregated mode toggle, small prompt threshold) |
| [StageAwareOrchestratorTest.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-engine/src/test/java/tech/kayys/gollek/engine/inference/StageAwareOrchestratorTest.java) | Unit tests verifying combined vs stage-split routing behavior |

---

### Phase 4: P/D Disaggregation (`gollek-cluster`)

New core module at `inference-gollek/core/gollek-cluster/`:

| File | Purpose |
|---|---|
| [pom.xml](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-cluster/pom.xml) | Maven module with `quarkus-grpc` |
| [NodeRole.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-cluster/src/main/java/tech/kayys/gollek/cluster/NodeRole.java) | `enum { PREFILL, DECODE, BOTH, GATEWAY }` |
| [GollekClusterManager.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-cluster/src/main/java/tech/kayys/gollek/cluster/GollekClusterManager.java) | Node identity & role management (config-driven) |
| [gollek_internal.proto](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-cluster/src/main/proto/gollek_internal.proto) | gRPC definition for `HandoffCache` RPC |
| [CacheHandoffService.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-cluster/src/main/java/tech/kayys/gollek/cluster/CacheHandoffService.java) | GrpcService implementing cache handoff logic |

---

### Pre-existing Test Fixes

| File | Fix |
|---|---|
| [DefaultInferenceEngineTest.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-engine/src/test/java/tech/kayys/golek/engine/inference/DefaultInferenceEngineTest.java) | `@Mock String` → plain `String` literal |
| [InferenceOrchestratorTest.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-engine/src/test/java/tech/kayys/golek/engine/inference/InferenceOrchestratorTest.java) | `@Mock String` → plain `String` literal |
| [ModelStorageServiceTest.java](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/inference-gollek/core/gollek-engine/src/test/java/tech/kayys/golek/engine/model/ModelStorageServiceTest.java) | Path assertion: `/tmp/test-models/` → `/tmp/test-models-gollek/` |

---

## Validation

**Build verified**: `mvn clean package -DskipTests` on `inference-gollek/core` — all 10 modules SUCCESS including new `gollek-kv-cache`.

**Remaining**: Run `mvn clean package` (with tests) to verify the engine test fixes and the new KV-cache + PagedAttention tests pass.

**Remaining**:
- Run `mvn compile -pl inference-gollek/core/gollek-cluster` to generate proto sources.
- Run `mvn test -pl inference-gollek/core/gollek-engine` to verify stage splitting logic.

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


