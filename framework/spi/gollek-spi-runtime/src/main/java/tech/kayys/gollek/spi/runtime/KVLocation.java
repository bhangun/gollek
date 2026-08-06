/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

/**
 * Represents the physical or logical location of a KV cache shard in the cluster.
 *
 * @param nodeId   the unique ID of the node holding the shard
 * @param address  the network address (IP:Port) for fetching the shard
 * @param tier     the storage tier (RAM, GPU, DISK, etc.)
 * @param latencyMs estimated latency to access this location
 */
public record KVLocation(
    String nodeId,
    String address,
    StorageTier tier,
    long latencyMs
) {
    /**
     * Represents a shard hosted in the local node's RAM.
     */
    public static KVLocation local(String address) {
        return new KVLocation("local", address, StorageTier.RAM, 0L);
    }
}
