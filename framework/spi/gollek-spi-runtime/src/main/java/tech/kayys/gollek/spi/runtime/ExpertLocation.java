/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

/**
 * Information about the physical or logical location of an MoE expert shard.
 * 
 * @param expertId  unique ID of the expert (e.g. "expert_42")
 * @param nodeId    ID of the node hosting the expert
 * @param address   network address (IP:Port) for remote access
 * @param tier      storage tier (RAM, GPU, DISK)
 */
public record ExpertLocation(
    String expertId,
    String nodeId,
    String address,
    StorageTier tier
) {
    /**
     * Creates a local expert location.
     */
    public static ExpertLocation local(String expertId) {
        return new ExpertLocation(expertId, "local", "localhost", StorageTier.GPU);
    }

    /**
     * Creates a remote expert location.
     */
    public static ExpertLocation remote(String expertId, String nodeId, String address) {
        return new ExpertLocation(expertId, nodeId, address, StorageTier.REMOTE);
    }
}
