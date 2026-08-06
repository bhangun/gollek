/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

/**
 * Represents a cluster node or runtime location where an {@link ExecutionProvider}
 * is hosted. This information is used for regional routing and data locality.
 *
 * @param nodeId unique identifier for the node
 * @param region geographic or logical region (e.g. "us-east-1", "local-hpc")
 * @param load   current utilization of the node (0.0 to 1.0)
 */
public record NodeInfo(
        String nodeId,
        String region,
        double load
) {
    /**
     * Represents a default local node when cluster information is unavailable.
     */
    public static final NodeInfo LOCAL = new NodeInfo("localhost", "local", 0.0);
}
