/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.lang.foreign.MemorySegment;
import java.util.Map;

/**
 * Interface for merging and weighting logit tensors from multiple models.
 * Essential for ensemble inference and speculative verification.
 */
public interface LogitMixer {

    /**
     * Merges multiple logit segments into a single output probability distribution.
     * 
     * @param logits  mapping of model/provider ID to their output logit segments
     * @param weights relative weights for each model's contribution
     * @param output  the target segment to write the merged results
     */
    void mix(Map<String, MemorySegment> logits, Map<String, Double> weights, MemorySegment output);

    /**
     * Normalizes weights to ensure they sum to 1.0.
     */
    default Map<String, Double> normalize(Map<String, Double> weights) {
        double sum = weights.values().stream().mapToDouble(Double::doubleValue).sum();
        if (sum <= 0) return weights;
        
        return weights.entrySet().stream()
            .collect(java.util.stream.Collectors.toMap(
                Map.Entry::getKey,
                e -> e.getValue() / sum
            ));
    }
}
