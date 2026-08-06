/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

/**
 * Control plane interface for the cluster-wide KV memory fabric.
 * Acts as a distributed directory for locating and registering token memory shards.
 */
public interface KVDirectory {

    /**
     * Locates all physical locations of a specific KV shard.
     *
     * @param key the unique identifier of the KV shard
     * @return a future resolving to the list of available locations, ordered by latency
     */
    CompletableFuture<List<KVLocation>> locate(KVKey key);

    /**
     * Registers a new shard location in the global directory.
     *
     * @param key      the unique identifier of the KV shard
     * @param location the location details to register
     */
    CompletableFuture<Void> register(KVKey key, KVLocation location);

    /**
     * Locates all physical locations of a specific MoE expert.
     */
    CompletableFuture<List<ExpertLocation>> locateExperts(String modelId, String expertId);

    /**
     * Registers a new expert location in the global directory.
     */
    CompletableFuture<Void> registerExpert(String modelId, ExpertLocation location);

    /**
     * Deregisters a location (e.g. node shutdown or cache eviction).
     *
     * @param key    the unique identifier of the KV shard
     * @param nodeId the ID of the node being deregistered
     */
    CompletableFuture<Void> evict(KVKey key, String nodeId);

    /**
     * Finds the single best location for a KV shard based on current load and latency.
     *
     * @param key the unique identifier of the KV shard
     * @return an optional containing the optimal location if found
     */
    default Optional<KVLocation> findOptimal(KVKey key) {
        try {
            List<KVLocation> locations = locate(key).get();
            return locations.isEmpty() ? Optional.empty() : Optional.of(locations.get(0));
        } catch (Exception e) {
            return Optional.empty();
        }
    }
}
