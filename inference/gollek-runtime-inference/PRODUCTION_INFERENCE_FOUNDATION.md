# Production Inference Foundation — Implementation Summary

## Overview

This document summarizes the implementation of **Phase 1: Foundation** components for production-ready LLM inference serving in Gollek. These components form the critical path for competing with vLLM, TGI, and other production inference engines.

## Components Implemented

### 1. PagedKVCache (`PagedKVCache.java`)

**Location:** `gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/kv/PagedKVCache.java`

**What it is:** Production-grade vLLM-style paged KV cache implementation using JDK 25 FFM (Foreign Function & Memory) API for off-heap native memory management.

**Key Features:**
- ✅ **Block Pool Management** — Pre-allocated native memory blocks eliminate fragmentation
- ✅ **Logical-to-Physical Mapping** — Block tables allow scattered memory access for attention kernels
- ✅ **Per-Sequence Isolation** — Each request gets its own `SequenceKVCache` with independent block table
- ✅ **Prefix Caching Support** — Snapshot/restore for shared prefix reuse
- ✅ **Multi-Tenant Tracking** — Per-sequence tenant ID for quota enforcement
- ✅ **FFM Native Memory** — Uses `Arena` for efficient off-heap allocation
- ✅ **OOM Protection** — Allocates from fixed pool, fails gracefully when full

**Memory Layout:**
```
Block Pool (pre-allocated in native memory)
├── Block 0: [K page | V page]  ← pageSize * numHeads * headDim * dtypeSize bytes
├── Block 1: [K page | V page]
├── Block 2: [K page | V page]
└── ...

Sequence A: blockTable = [0, 2, 5]  ← logical token order → physical blocks
Sequence B: blockTable = [1, 3, 4]
```

**API Example:**
```java
PagedKVCache cache = PagedKVCache.builder()
    .numBlocks(1024)
    .pageSize(16)
    .numLayers(32)
    .numHeads(32)
    .headDim(128)
    .dtype(DType.FLOAT16)
    .arena(Arena.ofAuto())
    .build();

// Create per-sequence cache
PagedKVCache.SequenceKVCache seqCache = cache.createSequenceCache("req-123", "tenant-1");

// Append K/V tensors during decode
seqCache.append(layer, kTensor, vTensor);

// Get block table for attention kernel
int[] blockTable = seqCache.getBlockTable(layer);

// Free when request completes
cache.freeSequence("req-123");
```

**Statistics Available:**
- `usedBlocks()` — Currently allocated blocks
- `availableBlocks()` — Free blocks remaining
- `memoryUtilization()` — Fraction of pool in use (0.0 to 1.0)
- `totalAppends()` — Total KV append operations
- `prefixCacheHits()` — Snapshot/reuse operations

---

### 2. KVCacheManager (`KVCacheManager.java`)

**Location:** `gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/kv/KVCacheManager.java`

**What it is:** Centralized manager for multiple KV cache instances with OOM protection, eviction policies, and tenant quota enforcement.

**Key Features:**
- ✅ **Multi-Model Support** — Different models get different KV cache configurations
- ✅ **LRU/FIFO Eviction** — Automatically evict sequences when memory pressure is high
- ✅ **Tenant Quotas** — Per-tenant block limits prevent noisy neighbor problems
- ✅ **Memory Pressure Monitoring** — Track utilization and trigger eviction at threshold
- ✅ **LRU Tracking** — Per-sequence access time tracking for intelligent eviction

**API Example:**
```java
KVCacheManager manager = KVCacheManager.builder()
    .globalMaxBlocks(4096)
    .evictionPolicy(EvictionPolicy.LRU)
    .evictionThreshold(0.85)  // Evict when 85% full
    .build();

// Register model's cache
manager.registerCache("llama-3-70b", pagedKVCache);

// Set tenant quota
manager.setTenantQuota("enterprise-1", 512);  // Max 512 blocks

// Create sequence (respects quotas)
PagedKVCache.SequenceKVCache seqCache = manager.createSequence(
    "llama-3-70b", "req-123", "enterprise-1");

// Check memory pressure before scheduling
if (manager.isMemoryPressureHigh("llama-3-70b")) {
    manager.evict("llama-3-70b");  // Evict LRU sequence
}

// Free sequence
manager.freeSequence("llama-3-70b", "req-123");
```

**Eviction Policies:**
- **LRU** — Evict sequence with oldest last access time
- **FIFO** — Evict sequence created earliest

---

### 3. PagedAttentionKernel (`PagedAttentionKernel.java`)

**Location:** `gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/kv/PagedAttentionKernel.java`

**What it is:** Interface defining the contract for attention kernels that operate on paged KV caches. Instead of requiring contiguous K/V tensors, these kernels read directly from scattered block memory via block tables.

**Key Operations:**
- `forward()` — Standard paged attention with block table
- `flashAttention()` — Tile-based flash attention with paged blocks
- `groupedQueryAttention()` — GQA support (LLaMA 2/3 style)

**Implementation Note:** Concrete implementations should use FFM to read directly from native memory segments in `PagedKVCache` blocks. For GPU backends, the block table is uploaded to GPU memory and the attention kernel uses indirect indexing.

---

### 4. ContinuousBatchScheduler (Enhanced)

**Location:** `gollek-runtime-inference/src/main/java/tech/kayys/gollek/runtime/inference/batch/ContinuousBatchScheduler.java`

**What Changed:** Enhanced the existing skeleton with full production integration:

**New Features:**
- ✅ **KV Cache Integration** — Automatically creates `SequenceKVCache` per request
- ✅ **Memory-Aware Admission** — Rejects requests when KV cache is full
- ✅ **Request Cancellation** — Cancel in-flight requests and reclaim blocks
- ✅ **Graceful Shutdown** — Drain active requests before stopping
- ✅ **Comprehensive Metrics** — Track processed, rejected, cancelled requests
- ✅ **Builder Pattern** — Clean configuration API
- ✅ **Virtual Threads** — Runs on JDK 25 virtual threads for massive concurrency

**Scheduler Loop:**
```
1. Drain new requests from queue (respecting memory limits)
2. Create KV cache for each admitted request
3. Execute one decode step for all active requests
4. Remove finished requests and free their KV cache
5. Repeat until shutdown requested
```

**API Example:**
```java
ContinuousBatchScheduler scheduler = ContinuousBatchScheduler.builder()
    .maxBatchSize(128)
    .modelId("llama-3-70b")
    .kvCacheManager(cacheManager)
    .pagedAttention(pagedAttentionKernel)
    .build();

scheduler.start(executor);

// Submit requests
scheduler.submit(batchRequest);

// Cancel if needed
scheduler.cancelRequest("req-123");

// Graceful shutdown
scheduler.stop();
scheduler.awaitTermination(5000);
```

**Metrics Available:**
- `getTotalBatchedRequests()` — Total requests processed
- `getTotalTokensGenerated()` — Total tokens generated
- `getTotalRejected()` — Requests rejected due to memory pressure
- `getTotalCancelled()` — Requests cancelled mid-flight
- `getAvgTokensPerRequest()` — Average output length

---

## Architecture Integration

### How Components Fit Together

```
┌─────────────────────────────────────────────────────────────┐
│                    REST API Layer                           │
│              (InferenceResource.java)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              ContinuousBatchScheduler                       │
│  - Admits requests from queue                               │
│  - Checks memory pressure via KVCacheManager                │
│  - Creates per-request SequenceKVCache                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  KVCacheManager                             │
│  - Tracks per-model cache usage                             │
│  - Enforces tenant quotas                                   │
│  - Evicts sequences when memory pressure high               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    PagedKVCache                             │
│  - Manages block pool (FFM native memory)                   │
│  - Allocates/frees blocks per sequence                      │
│  - Maintains block tables per layer                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 PagedAttentionKernel                        │
│  - Reads K/V from scattered blocks via block table          │
│  - Computes attention without contiguous memory             │
│  - Supports flash attention, GQA, etc.                      │
└─────────────────────────────────────────────────────────────┘
```

### Request Lifecycle

```
1. Client submits request via REST API
2. Request enters ContinuousBatchScheduler queue
3. Scheduler checks KVCacheManager.isMemoryPressureHigh()
   - If HIGH: reject request (OOM protection)
   - If OK: admit request
4. KVCacheManager.createSequence() allocates SequenceKVCache
5. Each decode iteration:
   a. Executor calls PagedAttentionKernel.forward()
   b. Kernel reads K/V from PagedKVCache blocks via block table
   c. New token sampled and appended to SequenceKVCache
   d. Token streamed to client (if streaming enabled)
6. Request finishes (EOS token or maxTokens reached)
7. KVCacheManager.freeSequence() reclaims blocks
8. Metrics updated
```

---

## Next Steps (Phase 1 Remaining)

### 1.3 GPU Memory Manager

Build arena-based allocator with:
- Pre-allocation at startup
- Defragmentation when sequences finish
- OOM prevention via hard limits
- Memory pooling for tensor reuse

### 1.4 Model Hot-Swap Registry

Implement:
- Versioned model handles
- Load/unload without restart
- Active model tracking
- Zero-downtime updates

### 1.5 Health Checks

Implement:
- Readiness probe (model loaded + warmed)
- Liveness probe (scheduler responsive)
- Startup probe (initialization complete)
- Metrics endpoint

---

## Performance Expectations

With these components properly integrated:

| Metric | Target | Notes |
|--------|--------|-------|
| **Concurrent Requests** | 128-256 | Depends on maxBatchSize |
| **KV Cache Utilization** | >85% | With LRU eviction |
| **Memory Efficiency** | 3-5x vs contiguous | From paged allocation |
| **Request Rejection Rate** | <5% | Under normal load |
| **Cancel Latency** | <10ms | Block reclamation |

---

## Testing Strategy

### Unit Tests
- `PagedKVCacheTest` — Block allocation, append, free
- `KVCacheManagerTest` — Multi-model, eviction, quotas
- `ContinuousBatchSchedulerTest` — Request lifecycle, metrics

### Integration Tests
- End-to-end request → KV cache → attention → response
- Memory pressure → eviction → new requests
- Graceful shutdown with active requests

### Load Tests
- Concurrent requests with varying lengths
- Memory pressure scenarios
- Tenant quota enforcement

---

## Migration from Legacy Code

The existing `DefaultBatchScheduler` in `gollek-engine` module remains unchanged and continues to work for non-LLM inference. The new `ContinuousBatchScheduler` in `gollek-runtime-inference` is specifically for LLM autoregressive generation with KV caching.

**When to use which:**
- `DefaultBatchScheduler` — Classification, embedding, non-autoregressive tasks
- `ContinuousBatchScheduler` — LLM text generation with KV caching

---

## References

- **vLLM Paper:** https://arxiv.org/abs/2309.06180
- **PagedAttention:** https://vllm.ai/
- **JDK 25 FFM API:** https://openjdk.org/jeps/454
- **Continuous Batching:** https://developer.nvidia.com/blog/mastering-llm-techniques-continuous-batching/

---

**Status:** Phase 1.1-1.2 Complete ✅ | 1.3-1.5 In Progress  
**Date:** 2026-04-12  
**Author:** Gollek Engineering Team
