/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

/**
 * Unique identifier for a KV cache segment in the global memory fabric.
 *
 * @param modelId    the ID of the model (e.g. "qwen2.5-7b-instruct")
 * @param prefixHash stable hash of the tokens (for prefix cache hits)
 * @param tenantId   optional tenant identifier for isolation
 * @param shardId    identifier for partitioned memory (e.g. "layer_0_15")
 * @param version    version of the cache segment
 */
public record KVKey(
    String modelId,
    String prefixHash,
    String tenantId,
    String shardId,
    long version
) {
    /**
     * Minimal key for shared system caches.
     */
    public static KVKey shared(String modelId, String prefixHash) {
        return new KVKey(modelId, prefixHash, "system", "all", 1L);
    }
}
