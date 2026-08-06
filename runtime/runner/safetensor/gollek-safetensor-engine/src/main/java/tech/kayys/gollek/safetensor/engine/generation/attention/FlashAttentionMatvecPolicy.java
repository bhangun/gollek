/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

record FlashAttentionMatvecPolicy(FlashAttentionMatvecOptions options) {

    FlashAttentionMatvecPolicy {
        if (options == null) {
            options = FlashAttentionMatvecOptions.defaults();
        }
    }

    static FlashAttentionMatvecPolicy from(FlashAttentionMatvecOptions options) {
        return new FlashAttentionMatvecPolicy(options);
    }

    boolean shouldUseMetalHalfMatvec(FlashAttentionModelPolicy modelPolicy, int outputDim) {
        if (options.disableMetalHalfMatvec()) {
            return false;
        }
        if (options.enableMetalHalfMatvec() != null) {
            return options.enableMetalHalfMatvec()
                    && outputDim <= options.metalHalfMatvecMaxOutput();
        }
        if (modelPolicy == null) {
            return false;
        }
        if (modelPolicy.preferNativeMetalBf16Linear()) {
            return outputDim <= options.metalHalfMatvecMaxOutput();
        }
        if (autoMetalHalfMatvecEnabled()) {
            return outputDim <= options.metalHalfMatvecMaxOutput()
                    && modelPolicy.metalHalfMatvecAutoCandidate();
        }
        return !autoMetalAttentionHalfMatvecExplicitlyDisabled()
                && outputDim <= options.metalHalfMatvecMaxOutput()
                && modelPolicy.compactAttentionMatvecCandidate();
    }

    boolean shouldUseMetalTransposedHalfMatvec(FlashAttentionModelPolicy modelPolicy, int outputDim) {
        if (options.disableMetalTransposedHalfMatvec()) {
            return false;
        }
        boolean enabled = options.enableMetalTransposedHalfMatvec() != null
                && options.enableMetalTransposedHalfMatvec();
        boolean eligiblePolicy = modelPolicy != null && modelPolicy.preferNativeMetalBf16Linear();
        if (!eligiblePolicy && !options.enableMetalTransposedHalfMatvecAll()) {
            return false;
        }
        return enabled && outputDim <= options.metalTransposedHalfMatvecMaxOutput();
    }

    boolean shouldUseMetalHalfTripleMatvec(int firstOutputDim, int secondOutputDim, int thirdOutputDim) {
        if (options.disableMetalMixedHalfLinearTripleMatvec()) {
            return false;
        }
        return firstOutputDim > 0
                && secondOutputDim > 0
                && thirdOutputDim > 0
                && firstOutputDim + secondOutputDim + thirdOutputDim
                        <= options.metalMixedHalfLinearTripleMatvecMaxOutput();
    }

    private boolean autoMetalHalfMatvecEnabled() {
        if (options.autoMetalAttentionHalfMatvec() != null) {
            return options.autoMetalAttentionHalfMatvec();
        }
        return options.autoMetalHalfMatvec() != null && options.autoMetalHalfMatvec();
    }

    private boolean autoMetalAttentionHalfMatvecExplicitlyDisabled() {
        if (options.autoMetalAttentionHalfMatvec() != null) {
            return !options.autoMetalAttentionHalfMatvec();
        }
        return options.autoMetalHalfMatvec() != null && !options.autoMetalHalfMatvec();
    }
}
