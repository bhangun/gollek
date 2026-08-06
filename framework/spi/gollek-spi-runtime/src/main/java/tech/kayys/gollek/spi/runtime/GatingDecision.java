/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.util.Map;

/**
 * Represents the result of a Mixture-of-Experts gating operation for a single token.
 * Defines which experts are active and their relative contribution weights.
 * 
 * @param expertWeights mapping of expert shard IDs to their activation weights (softmax output)
 * @param topK          the number of experts activated (usually 1 or 2)
 */
public record GatingDecision(
    Map<String, Float> expertWeights,
    int topK
) {
    /**
     * Top-1 gating result.
     */
    public static GatingDecision top1(String expertId, float weight) {
        return new GatingDecision(Map.of(expertId, weight), 1);
    }

    /**
     * Top-2 gating result.
     */
    public static GatingDecision top2(String e1, float w1, String e2, float w2) {
        return new GatingDecision(Map.of(e1, w1, e2, w2), 2);
    }
}
