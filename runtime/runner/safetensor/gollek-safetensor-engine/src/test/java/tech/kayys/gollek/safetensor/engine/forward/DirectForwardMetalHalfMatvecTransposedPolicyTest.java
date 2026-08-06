/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardMetalHalfMatvecTransposedPolicyTest {

    @Test
    void defaultsToNativeBf16LogitsOnly() {
        DirectForwardMetalHalfMatvecTransposedOptions options =
                DirectForwardMetalHalfMatvecTransposedOptions.defaults()
                        .withTransposedHalfMatvec(null, false, false, 4);
        DirectForwardMetalHalfMatvecTransposedPolicy policy =
                DirectForwardMetalHalfMatvecTransposedPolicy.from(options);

        assertTrue(policy.shouldUseMetalTransposedHalfMatvec(gemma4(), 4, "logits"));
        assertFalse(policy.shouldUseMetalTransposedHalfMatvec(gemma4(), 5, "logits"));
        assertFalse(policy.shouldUseMetalTransposedHalfMatvec(generic(), 4, "logits"));
        assertFalse(policy.shouldUseMetalTransposedHalfMatvec(gemma4(), 4, "q_proj"));
    }

    @Test
    void hiddenProjectorsRequireExplicitAllOptIn() {
        assertFalse(policy(true, false, false, 4)
                .shouldUseMetalTransposedHalfMatvec(generic(), 4, "q_proj"));
        assertTrue(policy(true, true, false, 4)
                .shouldUseMetalTransposedHalfMatvec(generic(), 4, "q_proj"));
    }

    @Test
    void genericLogitsRequireAllowAllAndExplicitEnable() {
        assertFalse(policy(null, true, false, 4)
                .shouldUseMetalTransposedHalfMatvec(generic(), 4, "logits"));
        assertFalse(policy(false, true, false, 4)
                .shouldUseMetalTransposedHalfMatvec(generic(), 4, "logits"));
        assertTrue(policy(true, true, false, 4)
                .shouldUseMetalTransposedHalfMatvec(generic(), 4, "logits"));
    }

    @Test
    void disableAndMaxOutputGateAllBranches() {
        assertFalse(policy(true, true, true, 4)
                .shouldUseMetalTransposedHalfMatvec(gemma4(), 4, "logits"));
        assertFalse(policy(true, true, false, 0)
                .shouldUseMetalTransposedHalfMatvec(gemma4(), 1, "logits"));
    }

    private static ModelConfigTraits generic() {
        return traits("llama", false);
    }

    private static ModelConfigTraits gemma4() {
        return traits("gemma4_text", true);
    }

    private static ModelConfigTraits traits(String modelType, boolean nativeBf16Matvec) {
        return new ModelConfigTraits(null, modelType, 0, 0, nativeBf16Matvec, false, false, false, false, false, false);
    }

    private static DirectForwardMetalHalfMatvecTransposedPolicy policy(
            Boolean enable,
            boolean allowAll,
            boolean disable,
            int maxOutput) {
        return DirectForwardMetalHalfMatvecTransposedPolicy.from(
                DirectForwardMetalHalfMatvecTransposedOptions.defaults()
                        .withTransposedHalfMatvec(enable, allowAll, disable, maxOutput));
    }
}
