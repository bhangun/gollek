/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.util.concurrent.CompletableFuture;

/**
 * Interface for per-token model selection and routing.
 * Unlike the IntelligentRouter (which routes requests), the TokenRouter
 * operates mid-inference to enable speculative decoding and model ensembles.
 */
public interface TokenRouter {

    /**
     * Decides which model(s) to use for the next N tokens.
     * 
     * @param context current context (tokens generated so far)
     * @param lastLogits optional logits from the previous token for confidence gating
     * @return a routing decision (single model, ensemble, or speculative pair)
     */
    CompletableFuture<RoutingDecision> route(UnifiedKVCache context, java.lang.foreign.MemorySegment lastLogits);

    /**
     * Feedback loop for the router to learn from the "winning" token or 
     * verification result.
     */
    default void update(String requestId, RoutingDecision decision, boolean wasAccepted) {
        // Default: no-op (learning logic goes here)
    }
}
