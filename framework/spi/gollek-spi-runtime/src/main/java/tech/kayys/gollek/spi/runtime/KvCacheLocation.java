/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

/**
 * Represents the location of a specific KV cache prefix on a provider.
 * This is used for cache-aware routing to minimize re-computation.
 *
 * @param providerId the ID of the provider hosting the cache
 * @param cacheKey   a unique identifier for the specific prefix/context
 * @param tokenCount the number of tokens covered by this cache entry
 */
public record KvCacheLocation(
        String providerId,
        String cacheKey,
        int tokenCount
) {
}
