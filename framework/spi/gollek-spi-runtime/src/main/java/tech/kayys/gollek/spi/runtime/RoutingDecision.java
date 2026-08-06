/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.util.Map;

/**
 * Represents a decision made by the Token Router for the next token generation.
 * Can trigger a single model, or a weighted ensemble.
 * 
 * @param modelWeights mapping of ExecutionProvider IDs to their relative weights (0.0 to 1.0)
 * @param strategy     the routing strategy (e.g. "ENSEMBLE", "SPECULATIVE", "GATED")
 */
public record RoutingDecision(
    Map<String, Double> modelWeights,
    String strategy
) {
    /**
     * Single model routing decision.
     */
    public static RoutingDecision single(String providerId) {
        return new RoutingDecision(Map.of(providerId, 1.0), "DIRECT");
    }

    /**
     * Predictive/Speculative routing where one model drafts and another verifies.
     */
    public static RoutingDecision speculative(String draftId, String targetId) {
        return new RoutingDecision(Map.of(draftId, 1.0, targetId, 0.0), "SPECULATIVE");
    }
}
