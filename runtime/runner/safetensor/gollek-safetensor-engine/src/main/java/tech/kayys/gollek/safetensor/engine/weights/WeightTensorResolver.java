/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.weights;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import java.util.Map;

/**
 * Resolves architecture-declared tensor-name candidates without embedding
 * model-family naming knowledge in the execution path.
 */
public final class WeightTensorResolver {
    private WeightTensorResolver() {
    }

    public static AccelTensor first(Map<String, AccelTensor> weights, Iterable<String> candidates,
            String... fallbackCandidates) {
        AccelTensor fromCandidates = first(weights, candidates);
        if (fromCandidates != null || fallbackCandidates == null || fallbackCandidates.length == 0) {
            return fromCandidates;
        }
        for (String fallback : fallbackCandidates) {
            AccelTensor tensor = tensor(weights, fallback);
            if (tensor != null) {
                return tensor;
            }
        }
        return null;
    }

    public static AccelTensor first(Map<String, AccelTensor> weights, Iterable<String> candidates) {
        if (weights == null || candidates == null) {
            return null;
        }
        for (String candidate : candidates) {
            AccelTensor tensor = tensor(weights, candidate);
            if (tensor != null) {
                return tensor;
            }
        }
        return null;
    }

    public static AccelTensor tensor(Map<String, AccelTensor> weights, String key) {
        return weights == null || key == null ? null : weights.get(key);
    }
}
