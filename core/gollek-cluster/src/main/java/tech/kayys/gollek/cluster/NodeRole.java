package tech.kayys.gollek.cluster;

import org.eclipse.microprofile.config.inject.ConfigProperty;

/**
 * Defines the role of this node in a disaggregated cluster.
 */
public enum NodeRole {
    /**
     * Compute-bound node optimized for processing prompts.
     * Generates the initial KV-cache and hands it off.
     */
    PREFILL,

    /**
     * Memory-bound node optimized for token generation.
     * Receives KV-cache and generates tokens autoregressively.
     */
    DECODE,

    /**
     * Handles both phases (default / standalone mode).
     */
    BOTH,

    /**
     * Gateway node that routes requests but performs no inference.
     */
    GATEWAY;

    /**
     * @return true if this node can handle prefill requests.
     */
    public boolean canPrefill() {
        return this == PREFILL || this == BOTH;
    }

    /**
     * @return true if this node can handle decode requests.
     */
    public boolean canDecode() {
        return this == DECODE || this == BOTH;
    }
}
