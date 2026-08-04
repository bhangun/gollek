/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardMetalHalfMatvecLogitsPolicyTest {

    @Test
    void logitsMpsMatvecRequiresExplicitEnableAndRejectsNativeBf16() {
        DirectForwardMetalHalfMatvecLogitsOptions options =
                DirectForwardMetalHalfMatvecLogitsOptions.defaults().withLogitsMpsMatvec(true, false, 10, 4);
        DirectForwardMetalHalfMatvecLogitsPolicy policy =
                DirectForwardMetalHalfMatvecLogitsPolicy.from(options);

        assertTrue(policy.shouldUseMetalLogitsMpsMatvec(generic(), 10, 4, "logits"));
        assertFalse(policy.shouldUseMetalLogitsMpsMatvec(generic(), 9, 4, "logits"));
        assertFalse(policy.shouldUseMetalLogitsMpsMatvec(generic(), 10, 5, "logits"));
        assertFalse(policy.shouldUseMetalLogitsMpsMatvec(generic(), 10, 4, "q_proj"));
        assertFalse(policy.shouldUseMetalLogitsMpsMatvec(gemma4(), 10, 4, "logits"));
        assertFalse(DirectForwardMetalHalfMatvecLogitsPolicy.from(
                options.withLogitsMpsMatvec(true, true, 10, 4))
                .shouldUseMetalLogitsMpsMatvec(generic(), 10, 4, "logits"));
    }

    @Test
    void familySpecificMaxOutputsOnlyApplyToLogits() {
        DirectForwardMetalHalfMatvecLogitsOptions options =
                DirectForwardMetalHalfMatvecLogitsOptions.defaults().withLogitsMaxOutputs(4, 5, 6);
        DirectForwardMetalHalfMatvecLogitsPolicy policy =
                DirectForwardMetalHalfMatvecLogitsPolicy.from(options);

        assertEquals(4, policy.metalHalfMatvecMaxOutput(gemma4(), "logits", 99));
        assertEquals(5, policy.metalHalfMatvecMaxOutput(gemma3(), "logits", 99));
        assertEquals(6, policy.metalHalfMatvecMaxOutput(qwen(), "logits", 99));
        assertEquals(99, policy.metalHalfMatvecMaxOutput(generic(), "logits", 99));
        assertEquals(99, policy.metalHalfMatvecMaxOutput(gemma4(), "q_proj", 99));
    }

    @Test
    void gemma3LogitsMatvecRequiresExplicitEnableAndDisableWins() {
        DirectForwardMetalHalfMatvecLogitsOptions enabled =
                DirectForwardMetalHalfMatvecLogitsOptions.defaults()
                        .withLogitsMaxOutputs(4, 4, 4)
                        .withGemma3Logits(true, false);
        DirectForwardMetalHalfMatvecLogitsPolicy policy =
                DirectForwardMetalHalfMatvecLogitsPolicy.from(enabled);

        assertTrue(policy.shouldUseGemma3LogitsMetalHalfMatvec(gemma3(), 4, "logits", 4));
        assertFalse(policy.shouldUseGemma3LogitsMetalHalfMatvec(gemma3(), 5, "logits", 4));
        assertFalse(policy.shouldUseGemma3LogitsMetalHalfMatvec(gemma3(), 4, "q_proj", 4));
        assertFalse(DirectForwardMetalHalfMatvecLogitsPolicy.from(DirectForwardMetalHalfMatvecLogitsOptions.defaults()
                .withLogitsMaxOutputs(4, 4, 4)
                .withGemma3Logits(null, false))
                .shouldUseGemma3LogitsMetalHalfMatvec(gemma3(), 4, "logits", 4));
        assertFalse(DirectForwardMetalHalfMatvecLogitsPolicy.from(enabled.withGemma3Logits(true, true))
                .shouldUseGemma3LogitsMetalHalfMatvec(gemma3(), 4, "logits", 4));
    }

    private static ModelConfigTraits generic() {
        return traits("llama", false, false, false);
    }

    private static ModelConfigTraits gemma4() {
        return traits("gemma4_text", true, false, false);
    }

    private static ModelConfigTraits gemma3() {
        return traits("gemma3_text", false, true, false);
    }

    private static ModelConfigTraits qwen() {
        return traits("qwen2", false, false, true);
    }

    private static ModelConfigTraits traits(
            String modelType,
            boolean gemma4Text,
            boolean gemma3Text,
            boolean qwenText) {
        return new ModelConfigTraits(null, modelType, 0, 0, gemma4Text, gemma3Text, qwenText, false, false, false, false);
    }
}
