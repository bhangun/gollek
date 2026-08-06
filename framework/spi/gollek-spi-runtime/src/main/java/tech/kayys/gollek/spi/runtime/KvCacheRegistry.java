/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.util.Optional;

/**
 * Registry to track and manage the locations of KV caches across
 * providers. This is used by the Intelligent Router to optimize
 * inference by routing to providers that already have relevant
 * context cached.
 */
public interface KvCacheRegistry {

    /**
     * Find a KV cache location for a given prefix hash.
     *
     * @param prefixHash the hash representing the prefix/context
     * @return an optional containing the location of the cache
     */
    Optional<KvCacheLocation> find(String prefixHash);

    /**
     * Register a new KV cache location.
     *
     * @param prefixHash the hash representing the prefix/context
     * @param location   the location details of the cache
     */
    void register(String prefixHash, KvCacheLocation location);
}
