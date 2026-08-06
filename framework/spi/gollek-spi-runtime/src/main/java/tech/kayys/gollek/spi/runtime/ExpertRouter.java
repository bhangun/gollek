/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.CompletableFuture;

/**
 * Interface for Mixture-of-Experts (MoE) gating logic.
 * Determines which experts are responsible for processing the current token activation.
 */
public interface ExpertRouter {

    /**
     * Routes a single token activation to its Top-K experts.
     * 
     * @param inputActivation the hidden state tensor before the MoE layer
     * @param topK            the number of experts to select (e.g. 1 or 2)
     * @return a future result containing the expert IDs and weights
     */
    CompletableFuture<GatingDecision> route(MemorySegment inputActivation, int topK);

    /**
     * Decides whether an expert should be executed locally or on a remote node
     * based on the ExpertLocation directory.
     */
    default boolean shouldRouteRemote(String expertId, KVDirectory directory) {
        // Implementation: check if expertId exists in local registry
        return false; 
    }
}
