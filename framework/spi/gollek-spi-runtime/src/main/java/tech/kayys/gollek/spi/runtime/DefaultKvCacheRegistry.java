/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import jakarta.enterprise.context.ApplicationScoped;

/**
 * Default local-only implementation of the KvCacheRegistry.
 * Tracks KV cache locations in-memory for the current application instance.
 */
@ApplicationScoped
public class DefaultKvCacheRegistry implements KvCacheRegistry {

    private final Map<String, KvCacheLocation> registry = new ConcurrentHashMap<>();

    @Override
    public Optional<KvCacheLocation> find(String prefixHash) {
        if (prefixHash == null) {
            return Optional.empty();
        }
        return Optional.ofNullable(registry.get(prefixHash));
    }

    @Override
    public void register(String prefixHash, KvCacheLocation location) {
        if (prefixHash != null && location != null) {
            registry.put(prefixHash, location);
        }
    }
}
