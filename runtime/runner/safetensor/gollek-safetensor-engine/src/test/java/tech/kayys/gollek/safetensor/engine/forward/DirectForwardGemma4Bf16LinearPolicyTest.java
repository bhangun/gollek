/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardNativeBf16Bf16LinearPolicyTest {

    @Test
    void metalBf16DefaultsOnOnlyForNativeBf16Text() {
        DirectForwardNativeBf16Bf16LinearPolicy defaults = policy(DirectForwardNativeBf16Bf16LinearOptions.defaults());

        assertTrue(defaults.allowMetalBf16Linear(gemma4()));
        assertTrue(defaults.preferNativeMetalBf16Linear(gemma4()));
        assertFalse(defaults.allowMetalBf16Linear(generic()));
        assertFalse(defaults.preferNativeMetalBf16Linear(generic()));
    }

    @Test
    void metalBf16DisableAndExplicitRejectWin() {
        assertFalse(policy(DirectForwardNativeBf16Bf16LinearOptions.defaults()
                .withMetalBf16(null, true))
                .allowMetalBf16Linear(gemma4()));
        assertFalse(policy(DirectForwardNativeBf16Bf16LinearOptions.defaults()
                .withMetalBf16(false, false))
                .preferNativeMetalBf16Linear(gemma4()));
    }

    @Test
    void globalBf16ToF16OverrideWinsBeforeProjectionDefaults() {
        DirectForwardNativeBf16Bf16LinearOptions enabled =
                DirectForwardNativeBf16Bf16LinearOptions.defaults().withBf16ToF16(true, false);
        DirectForwardNativeBf16Bf16LinearPolicy policy = policy(enabled);

        assertTrue(policy.allowBf16ToF16Linear(gemma4(), "ffn_gate"));
        assertTrue(policy.allowBf16ToF16Linear(gemma4(), "logits"));
        assertFalse(policy(enabled.withBf16ToF16(true, true)).allowBf16ToF16Linear(gemma4(), "ffn_gate"));
    }

    @Test
    void projectionSpecificBf16ToF16RulesStayScoped() {
        assertTrue(policy(DirectForwardNativeBf16Bf16LinearOptions.defaults()
                .withBf16ToF16Ffn(true, false))
                .allowBf16ToF16Linear(gemma4(), "ffn_down"));
        assertFalse(policy(DirectForwardNativeBf16Bf16LinearOptions.defaults()
                .withBf16ToF16Ffn(true, false))
                .allowBf16ToF16Linear(gemma4(), "logits"));
        assertTrue(policy(DirectForwardNativeBf16Bf16LinearOptions.defaults()
                .withBf16ToF16Logits(true, false))
                .allowBf16ToF16Linear(gemma4(), "logits"));
        assertFalse(policy(DirectForwardNativeBf16Bf16LinearOptions.defaults()
                .withBf16ToF16Logits(true, true))
                .allowBf16ToF16Linear(gemma4(), "logits"));
    }

    @Test
    void rowsGateKeepsPrefillNativeAndAllowsDecodeLogitsOnly() {
        DirectForwardNativeBf16Bf16LinearPolicy options =
                policy(DirectForwardNativeBf16Bf16LinearOptions.defaults().withBf16ToF16Logits(true, false));

        assertFalse(options.allowBf16ToF16LinearForRows(2, gemma4(), "logits", true));
        assertFalse(options.allowBf16ToF16LinearForRows(1, gemma4(), "logits", false));
        assertTrue(options.allowBf16ToF16LinearForRows(1, gemma4(), "logits", true));
    }

    @Test
    void nonNativeBf16NeverUsesBf16ToF16Path() {
        assertFalse(policy(DirectForwardNativeBf16Bf16LinearOptions.defaults()
                .withBf16ToF16(true, false))
                .allowBf16ToF16Linear(generic(), "ffn_gate"));
    }

    private static ModelConfigTraits gemma4() {
        return traits(true);
    }

    private static ModelConfigTraits generic() {
        return traits(false);
    }

    private static ModelConfigTraits traits(boolean nativeBf16Matvec) {
        return new ModelConfigTraits(null, nativeBf16Matvec ? "gemma4_text" : "llama",
                0, 0, nativeBf16Matvec, false, false, false, false, false, false);
    }

    private static DirectForwardNativeBf16Bf16LinearPolicy policy(
            DirectForwardNativeBf16Bf16LinearOptions options) {
        return DirectForwardNativeBf16Bf16LinearPolicy.from(options);
    }
}
