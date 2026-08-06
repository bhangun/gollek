/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.multiplySaturating;

import java.util.Objects;

record DirectForwardLinearCacheRoutingPolicy(DirectForwardLinearCacheOptions options) {

    DirectForwardLinearCacheRoutingPolicy {
        options = Objects.requireNonNull(options, "options");
    }

    static DirectForwardLinearCacheRoutingPolicy from(DirectForwardLinearCacheOptions options) {
        return new DirectForwardLinearCacheRoutingPolicy(options);
    }

    boolean shouldCacheFfnDownHalfWeight(
            String profileKey,
            boolean singleTokenCandidate,
            long weightDequantizedBytes,
            int numHiddenLayers) {
        if (!isFfnDownHalfCacheProfile(profileKey)) {
            return false;
        }
        if (!singleTokenCandidate) {
            return false;
        }
        if (options.ffnDownLargeHalfCachePerTensorMaxBytes() <= 0L
                || weightDequantizedBytes > options.ffnDownLargeHalfCachePerTensorMaxBytes()) {
            return false;
        }
        if (options.ffnDownLargeHalfCacheTotalMaxBytes() <= 0L) {
            return false;
        }
        long estimatedModelBytes = multiplySaturating(weightDequantizedBytes, numHiddenLayers);
        return estimatedModelBytes <= options.ffnDownLargeHalfCacheTotalMaxBytes();
    }

    boolean shouldCacheLogitsLargeHalfWeight(
            String profileKey,
            boolean singleTokenCandidate,
            long weightDequantizedBytes) {
        if (!isLogitsLargeHalfCacheProfile(profileKey)) {
            return false;
        }
        if (!singleTokenCandidate) {
            return false;
        }
        return options.logitsLargeHalfCacheMaxBytes() > 0L
                && weightDequantizedBytes <= options.logitsLargeHalfCacheMaxBytes();
    }

    boolean isFfnDownHalfCacheProfile(String profileKey) {
        return "ffn_down".equals(profileKey) || "ffn_down_nongated".equals(profileKey);
    }

    boolean isLogitsLargeHalfCacheProfile(String profileKey) {
        return "logits".equals(profileKey);
    }
}
