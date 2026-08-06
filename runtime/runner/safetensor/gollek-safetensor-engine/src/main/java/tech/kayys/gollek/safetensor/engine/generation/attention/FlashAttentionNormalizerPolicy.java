/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

record FlashAttentionNormalizerPolicy(FlashAttentionNormalizerOptions options) {

    FlashAttentionNormalizerPolicy {
        if (options == null) {
            options = FlashAttentionNormalizerOptions.defaults();
        }
    }

    static FlashAttentionNormalizerPolicy from(FlashAttentionNormalizerOptions options) {
        return new FlashAttentionNormalizerPolicy(options);
    }

    boolean shouldUseMetalPerHeadRmsNorm(FlashAttentionModelPolicy modelPolicy) {
        if (options.disableMetalPerHeadRmsNorm()) {
            return false;
        }
        String explicitValue = options.enableMetalPerHeadRmsNormValue();
        if (explicitValue != null && !explicitValue.isBlank()) {
            return Boolean.parseBoolean(explicitValue);
        }
        return modelPolicy != null && modelPolicy.preferMetalPerHeadRmsNorm();
    }
}
