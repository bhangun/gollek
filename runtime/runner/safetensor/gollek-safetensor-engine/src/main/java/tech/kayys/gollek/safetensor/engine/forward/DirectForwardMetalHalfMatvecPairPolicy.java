/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

import tech.kayys.alkhawarizm.spi.model.ModelConfig;

record DirectForwardMetalHalfMatvecPairPolicy(
        DirectForwardMetalHalfMatvecPairOptions options,
        DirectForwardMetalHalfMatvecAutoPolicy autoPolicy) {

    DirectForwardMetalHalfMatvecPairPolicy {
        options = Objects.requireNonNull(options, "options");
        autoPolicy = Objects.requireNonNull(autoPolicy, "autoPolicy");
    }

    static DirectForwardMetalHalfMatvecPairPolicy from(
            DirectForwardMetalHalfMatvecPairOptions options,
            DirectForwardMetalHalfMatvecCoreOptions coreOptions) {
        return new DirectForwardMetalHalfMatvecPairPolicy(
                options,
                DirectForwardMetalHalfMatvecAutoPolicy.from(coreOptions));
    }

    boolean shouldUseMetalHalfLinearPair(
            ModelConfigTraits traits,
            boolean multiRowLinearInput,
            boolean allowGemma4FusedHalfFfn) {
        if (!metalHalfLinearPairEnabled()) {
            return false;
        }
        if (options.enableMetalHalfLinearPair() != null) {
            return options.enableMetalHalfLinearPair();
        }
        if (traits.nativeBf16Matvec() || traits.perLayerInputEmbedding()) {
            return multiRowLinearInput || !DirectForwardFfnFastPathPolicy.isNativeBf16FusedHalfFfnAllowed();
        }
        return true;
    }

    boolean shouldUseMetalHalfMatvecPair(ModelConfigTraits traits, ModelConfig config, int outputDim) {
        if (options.disableMetalHalfMatvecPair()) {
            return false;
        }
        if (options.enableMetalHalfMatvecPair() != null) {
            return options.enableMetalHalfMatvecPair() && outputDim <= autoPolicy.defaultMaxOutput();
        }
        return autoPolicy.shouldUseMetalHalfMatvec(traits, config, outputDim);
    }

    private boolean metalHalfLinearPairEnabled() {
        if (options.disableMetalHalfLinearPair()) {
            return false;
        }
        if (options.enableMetalHalfLinearPair() != null) {
            return options.enableMetalHalfLinearPair();
        }
        return true;
    }
}
