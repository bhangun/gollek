/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

/**
 * Routing policy for native BF16 Metal linear projections.
 *
 * <p>This class is model-family agnostic. It routes based on the
 * {@link ModelConfigTraits#nativeBf16Matvec()} capability flag, which any
 * model may declare. No model identity (gemma4, gemma3, etc.) is referenced here.
 */
record DirectForwardNativeBf16LinearPolicy(DirectForwardNativeBf16LinearOptions options) {
    private static final String LOGITS_PROFILE_KEY = "logits";
    private static final String FFN_PROFILE_PREFIX = "ffn_";

    DirectForwardNativeBf16LinearPolicy {
        options = Objects.requireNonNull(options, "options");
    }

    static DirectForwardNativeBf16LinearPolicy from(DirectForwardNativeBf16LinearOptions options) {
        return new DirectForwardNativeBf16LinearPolicy(options);
    }

    boolean allowMetalBf16Linear(ModelConfigTraits traits) {
        if (!traits.nativeBf16Matvec() || options.disableMetalBf16Linear()) {
            return false;
        }
        if (options.enableMetalBf16Linear() != null) {
            return options.enableMetalBf16Linear();
        }
        return true;
    }

    boolean preferNativeMetalBf16Linear(ModelConfigTraits traits) {
        if (options.enableMetalBf16Linear() != null) {
            return options.enableMetalBf16Linear() && allowMetalBf16Linear(traits);
        }
        return allowMetalBf16Linear(traits);
    }

    boolean allowBf16ToF16Linear(ModelConfigTraits traits, String profileKey) {
        if (!traits.nativeBf16Matvec() || options.disableBf16ToF16Linear()) {
            return false;
        }
        if (LOGITS_PROFILE_KEY.equals(profileKey) && options.disableBf16ToF16LogitsLinear()) {
            return false;
        }
        if (options.enableBf16ToF16Linear() != null) {
            return options.enableBf16ToF16Linear();
        }
        if (LOGITS_PROFILE_KEY.equals(profileKey)) {
            if (options.enableBf16ToF16LogitsLinear() != null) {
                return options.enableBf16ToF16LogitsLinear();
            }
            return false;
        }
        if (options.enableBf16ToF16FfnLinear() != null) {
            return options.enableBf16ToF16FfnLinear();
        }
        return false;
    }

    boolean allowBf16ToF16LinearForRows(
            long rows,
            ModelConfigTraits traits,
            String profileKey,
            boolean decodeLogitsPhase) {
        // Keep prompt prefill on native BF16. Cached F16 conversion is useful
        // for repeated decode matvecs, but paying it before first token hurts TTFT.
        if (rows != 1L || !allowBf16ToF16Linear(traits, profileKey)) {
            return false;
        }
        return !LOGITS_PROFILE_KEY.equals(profileKey) || decodeLogitsPhase;
    }
}
