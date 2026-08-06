/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.util.List;

/**
 * Represents a batch of tokens proposed by a speculative "draft" model.
 * These tokens are unverified and must be validated by a larger target model.
 * 
 * @param draftTokens  the sequence of token IDs proposed by the draft model
 * @param confidences  confidence scores (probabilities) for each drafted token
 * @param requestId    the parent request ID
 */
public record SpeculativeBatch(
    List<Integer> draftTokens,
    List<Float> confidences,
    String requestId
) {
    /**
     * Size of the speculative window.
     */
    public int size() {
        return draftTokens.size();
    }

    /**
     * Truncates the batch if verification fails at a certain index.
     */
    public SpeculativeBatch truncate(int validCount) {
        return new SpeculativeBatch(
            draftTokens.subList(0, validCount),
            confidences.subList(0, validCount),
            requestId
        );
    }
}
