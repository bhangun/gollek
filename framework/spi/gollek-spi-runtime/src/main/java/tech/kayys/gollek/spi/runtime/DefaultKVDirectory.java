/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

import jakarta.enterprise.context.ApplicationScoped;

/**
 * Default local-first implementation of the KVDirectory.
 * Manages shard locations in-memory for the current cluster node.
 */
@ApplicationScoped
public class DefaultKVDirectory implements KVDirectory {

    private final Map<KVKey, List<KVLocation>> registry = new ConcurrentHashMap<>();
    private final Map<String, List<ExpertLocation>> expertRegistry = new ConcurrentHashMap<>();

    @Override
    public CompletableFuture<List<KVLocation>> locate(KVKey key) {
        List<KVLocation> locations = registry.getOrDefault(key, Collections.emptyList());
        return CompletableFuture.completedFuture(new ArrayList<>(locations));
    }

    @Override
    public CompletableFuture<Void> register(KVKey key, KVLocation location) {
        registry.computeIfAbsent(key, k -> new java.util.concurrent.CopyOnWriteArrayList<>()).add(location);
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<List<ExpertLocation>> locateExperts(String modelId, String expertId) {
        List<ExpertLocation> locations = expertRegistry.getOrDefault(expertId, Collections.emptyList());
        return CompletableFuture.completedFuture(new ArrayList<>(locations));
    }

    @Override
    public CompletableFuture<Void> registerExpert(String modelId, ExpertLocation location) {
        expertRegistry.computeIfAbsent(location.expertId(), k -> new java.util.concurrent.CopyOnWriteArrayList<>()).add(location);
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<Void> evict(KVKey key, String nodeId) {
        List<KVLocation> locations = registry.get(key);
        if (locations != null) {
            locations.removeIf(loc -> loc.nodeId().equals(nodeId));
        }
        return CompletableFuture.completedFuture(null);
    }
}
