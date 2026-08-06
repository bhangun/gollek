/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.spi.model.ModelConfig;

final class DirectForwardMetalHalfMatvecPolicy {
    private static final DirectForwardMetalHalfMatvecRoutingPolicy POLICY =
            DirectForwardMetalHalfMatvecRoutingPolicy.from(
                    DirectForwardMetalHalfMatvecOptions.fromSystemProperties());

    private DirectForwardMetalHalfMatvecPolicy() {
    }

    static boolean shouldUseMetalHalfLinearPair(
            ModelConfigTraits traits,
            boolean multiRowLinearInput,
            boolean allowGemma4FusedHalfFfn) {
        return POLICY.shouldUseMetalHalfLinearPair(traits, multiRowLinearInput, allowGemma4FusedHalfFfn);
    }

    static boolean shouldUseMetalHalfMatvec(
            ModelConfigTraits traits,
            ModelConfig config,
            int outputDim,
            String profileKey) {
        return POLICY.shouldUseMetalHalfMatvec(traits, config, outputDim, profileKey);
    }

    static boolean shouldUseMetalHalfMatvecPair(ModelConfigTraits traits, ModelConfig config, int outputDim) {
        return POLICY.shouldUseMetalHalfMatvecPair(traits, config, outputDim);
    }

    static boolean shouldUseMetalLogitsMpsMatvec(
            ModelConfigTraits traits,
            int outputDim,
            int inputDim,
            String profileKey) {
        return POLICY.shouldUseMetalLogitsMpsMatvec(traits, outputDim, inputDim, profileKey);
    }

    static boolean shouldUseMetalTransposedHalfMatvec(
            ModelConfigTraits traits,
            int outputDim,
            String profileKey) {
        return POLICY.shouldUseMetalTransposedHalfMatvec(traits, outputDim, profileKey);
    }
}
