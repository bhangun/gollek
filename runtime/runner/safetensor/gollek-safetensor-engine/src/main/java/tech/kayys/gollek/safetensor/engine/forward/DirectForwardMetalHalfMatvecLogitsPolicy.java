/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

record DirectForwardMetalHalfMatvecLogitsPolicy(DirectForwardMetalHalfMatvecLogitsOptions options) {
    private static final String LOGITS_PROFILE_KEY = "logits";

    DirectForwardMetalHalfMatvecLogitsPolicy {
        options = Objects.requireNonNull(options, "options");
    }

    static DirectForwardMetalHalfMatvecLogitsPolicy from(DirectForwardMetalHalfMatvecLogitsOptions options) {
        return new DirectForwardMetalHalfMatvecLogitsPolicy(options);
    }

    boolean shouldUseMetalLogitsMpsMatvec(
            ModelConfigTraits traits,
            int outputDim,
            int inputDim,
            String profileKey) {
        if (!isLogitsProfile(profileKey) || options.disableMetalLogitsMpsMatvec()) {
            return false;
        }
        if (!Boolean.TRUE.equals(options.enableMetalLogitsMpsMatvec())) {
            return false;
        }
        if (traits.nativeBf16Matvec()) {
            return false;
        }
        return outputDim >= options.metalLogitsMpsMatvecMinOutput()
                && (options.metalLogitsMpsMatvecMaxInput() <= 0
                || inputDim <= options.metalLogitsMpsMatvecMaxInput());
    }

    int metalHalfMatvecMaxOutput(ModelConfigTraits traits, String profileKey, int defaultMaxOutput) {
        if (!isLogitsProfile(profileKey)) {
            return defaultMaxOutput;
        }
        if (traits.nativeBf16Matvec()) {
            return options.nativeBf16LogitsMetalHalfMatvecMaxOutput();
        }
        if (traits.preferLargeLogitsMatvec() || traits.siluGated()) {
            return options.largeLogitsMetalHalfMatvecMaxOutput();
        }
        return defaultMaxOutput;
    }

    boolean shouldUseTurnAwarePromptBosLogitsMetalHalfMatvec(
            ModelConfigTraits traits,
            int outputDim,
            String profileKey,
            int maxOutput) {
        if (!isLogitsProfile(profileKey)
                || !traits.requiresTurnAwarePromptBos() // Formerly gemma3Text
                || options.disableTurnAwarePromptBosLogitsMetalHalfMatvec()) {
            return false;
        }
        if (maxOutput <= 0 || outputDim > maxOutput) {
            return false;
        }
        // M4 profiles currently prefer the default MPS matmul logits path for
        // FunctionGemma; keep this branch explicit so experiments cannot drift
        // into the default decode policy.
        return Boolean.TRUE.equals(options.enableTurnAwarePromptBosLogitsMetalHalfMatvec());
    }

    private static boolean isLogitsProfile(String profileKey) {
        return LOGITS_PROFILE_KEY.equals(profileKey);
    }
}
