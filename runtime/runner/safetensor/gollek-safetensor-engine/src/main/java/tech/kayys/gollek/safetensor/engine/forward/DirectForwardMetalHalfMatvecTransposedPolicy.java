/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

record DirectForwardMetalHalfMatvecTransposedPolicy(
        DirectForwardMetalHalfMatvecTransposedOptions options) {
    private static final String LOGITS_PROFILE_KEY = "logits";

    DirectForwardMetalHalfMatvecTransposedPolicy {
        options = Objects.requireNonNull(options, "options");
    }

    static DirectForwardMetalHalfMatvecTransposedPolicy from(
            DirectForwardMetalHalfMatvecTransposedOptions options) {
        return new DirectForwardMetalHalfMatvecTransposedPolicy(options);
    }

    boolean shouldUseMetalTransposedHalfMatvec(
            ModelConfigTraits traits,
            int outputDim,
            String profileKey) {
        if (options.disableMetalTransposedHalfMatvec()) {
            return false;
        }
        if (options.metalTransposedHalfMatvecMaxOutput() <= 0
                || outputDim > options.metalTransposedHalfMatvecMaxOutput()) {
            return false;
        }
        boolean explicitEnabled = Boolean.TRUE.equals(options.enableMetalTransposedHalfMatvec());
        boolean logitsProjection = LOGITS_PROFILE_KEY.equals(profileKey);
        if (!logitsProjection) {
            // Hidden projections stay behind a stronger opt-in because the
            // transposed-weight matvec kernel is still experimental there.
            return explicitEnabled && options.enableMetalTransposedHalfMatvecAll();
        }
        if (!traits.nativeBf16Matvec() && !options.enableMetalTransposedHalfMatvecAll()) {
            return false;
        }
        if (options.enableMetalTransposedHalfMatvec() != null) {
            return explicitEnabled;
        }
        return traits.nativeBf16Matvec();
    }
}
