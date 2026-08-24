/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

import tech.kayys.alkhawarizm.spi.model.ModelConfig;

record DirectForwardMetalHalfMatvecCorePolicy(
        DirectForwardMetalHalfMatvecCoreOptions options,
        DirectForwardMetalHalfMatvecAutoPolicy autoPolicy,
        DirectForwardMetalHalfMatvecLogitsPolicy logitsPolicy) {

    DirectForwardMetalHalfMatvecCorePolicy {
        options = Objects.requireNonNull(options, "options");
        autoPolicy = Objects.requireNonNull(autoPolicy, "autoPolicy");
        logitsPolicy = Objects.requireNonNull(logitsPolicy, "logitsPolicy");
    }

    static DirectForwardMetalHalfMatvecCorePolicy from(
            DirectForwardMetalHalfMatvecCoreOptions options,
            DirectForwardMetalHalfMatvecLogitsPolicy logitsPolicy) {
        return new DirectForwardMetalHalfMatvecCorePolicy(
                options,
                DirectForwardMetalHalfMatvecAutoPolicy.from(options),
                logitsPolicy);
    }

    boolean shouldUseMetalHalfMatvec(
            ModelConfigTraits traits,
            ModelConfig config,
            int outputDim,
            String profileKey) {
        if (options.disableMetalHalfMatvec()) {
            return false;
        }
        int maxOutput = logitsPolicy.metalHalfMatvecMaxOutput(
                traits,
                profileKey,
                options.metalHalfMatvecMaxOutput());
        if (logitsPolicy.shouldUseTurnAwarePromptBosLogitsMetalHalfMatvec(traits, outputDim, profileKey, maxOutput)) {
            return true;
        }
        if (options.enableMetalHalfMatvec() != null) {
            return options.enableMetalHalfMatvec() && maxOutput > 0 && outputDim <= maxOutput;
        }
        return autoPolicy.shouldUseMetalHalfMatvec(traits, config, outputDim, maxOutput);
    }
}
