/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

import tech.kayys.alkhawarizm.spi.model.ModelConfig;

record DirectForwardMetalHalfMatvecRoutingPolicy(
        DirectForwardMetalHalfMatvecLogitsPolicy logitsPolicy,
        DirectForwardMetalHalfMatvecCorePolicy corePolicy,
        DirectForwardMetalHalfMatvecPairPolicy pairPolicy,
        DirectForwardMetalHalfMatvecTransposedPolicy transposedPolicy) {

    DirectForwardMetalHalfMatvecRoutingPolicy {
        logitsPolicy = Objects.requireNonNull(logitsPolicy, "logitsPolicy");
        corePolicy = Objects.requireNonNull(corePolicy, "corePolicy");
        pairPolicy = Objects.requireNonNull(pairPolicy, "pairPolicy");
        transposedPolicy = Objects.requireNonNull(transposedPolicy, "transposedPolicy");
    }

    static DirectForwardMetalHalfMatvecRoutingPolicy from(DirectForwardMetalHalfMatvecOptions options) {
        DirectForwardMetalHalfMatvecOptions resolved =
                options == null ? DirectForwardMetalHalfMatvecOptions.defaults() : options;
        DirectForwardMetalHalfMatvecLogitsPolicy logits =
                DirectForwardMetalHalfMatvecLogitsPolicy.from(resolved.logitsOptions());
        return new DirectForwardMetalHalfMatvecRoutingPolicy(
                logits,
                DirectForwardMetalHalfMatvecCorePolicy.from(resolved.coreOptions(), logits),
                DirectForwardMetalHalfMatvecPairPolicy.from(resolved.pairOptions(), resolved.coreOptions()),
                DirectForwardMetalHalfMatvecTransposedPolicy.from(resolved.transposedOptions()));
    }

    boolean shouldUseMetalHalfLinearPair(
            ModelConfigTraits traits,
            boolean multiRowLinearInput,
            boolean allowGemma4FusedHalfFfn) {
        return pairPolicy.shouldUseMetalHalfLinearPair(traits, multiRowLinearInput, allowGemma4FusedHalfFfn);
    }

    boolean shouldUseMetalHalfMatvec(
            ModelConfigTraits traits,
            ModelConfig config,
            int outputDim,
            String profileKey) {
        return corePolicy.shouldUseMetalHalfMatvec(traits, config, outputDim, profileKey);
    }

    boolean shouldUseMetalHalfMatvecPair(ModelConfigTraits traits, ModelConfig config, int outputDim) {
        return pairPolicy.shouldUseMetalHalfMatvecPair(traits, config, outputDim);
    }

    boolean shouldUseMetalLogitsMpsMatvec(
            ModelConfigTraits traits,
            int outputDim,
            int inputDim,
            String profileKey) {
        return logitsPolicy.shouldUseMetalLogitsMpsMatvec(traits, outputDim, inputDim, profileKey);
    }

    boolean shouldUseMetalTransposedHalfMatvec(
            ModelConfigTraits traits,
            int outputDim,
            String profileKey) {
        return transposedPolicy.shouldUseMetalTransposedHalfMatvec(traits, outputDim, profileKey);
    }
}
