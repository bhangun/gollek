/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardMetalLinearRoutingPolicyTest {

    @Test
    void defaultsKeepExperimentalMetalLinearEnabled() {
        assertTrue(policy(DirectForwardMetalLinearOptions.defaults()).experimentalMetalLinearEnabled());
        assertFalse(policy(DirectForwardMetalLinearOptions.defaults()
                .withExperimentalMetalLinear(false))
                .experimentalMetalLinearEnabled());
    }

    @Test
    void genericBf16RequiresExplicitEnable() {
        ModelConfigTraits generic = traits(false);

        assertFalse(policy(DirectForwardMetalLinearOptions.defaults()).allowMetalBf16Linear(generic));
        assertTrue(policy(DirectForwardMetalLinearOptions.defaults()
                .withGenericBf16(true, false, false))
                .allowMetalBf16Linear(generic));
        assertFalse(policy(DirectForwardMetalLinearOptions.defaults()
                .withGenericBf16(true, true, false))
                .allowMetalBf16Linear(generic));
    }

    @Test
    void nativeBf16DefaultsOnUnlessDisabledOrExplicitlyRejected() {
        ModelConfigTraits gemma4 = traits(true);

        assertTrue(policy(DirectForwardMetalLinearOptions.defaults()).allowMetalBf16Linear(gemma4));
        assertFalse(policy(DirectForwardMetalLinearOptions.defaults()
                .withNativeBf16Bf16(null, true))
                .allowMetalBf16Linear(gemma4));
        assertFalse(policy(DirectForwardMetalLinearOptions.defaults()
                .withNativeBf16Bf16(false, false))
                .allowMetalBf16Linear(gemma4));
    }

    @Test
    void nativeBf16ToF16GlobalOverrideWinsBeforeProjectionSpecificDefaults() {
        ModelConfigTraits gemma4 = traits(true);

        assertFalse(policy(DirectForwardMetalLinearOptions.defaults())
                .allowNativeBf16Bf16ToF16Linear(gemma4, "ffn_gate"));
        assertFalse(policy(DirectForwardMetalLinearOptions.defaults())
                .allowNativeBf16Bf16ToF16Linear(gemma4, "logits"));
        assertTrue(policy(DirectForwardMetalLinearOptions.defaults()
                .withNativeBf16Bf16ToF16(true, false))
                .allowNativeBf16Bf16ToF16Linear(gemma4, "ffn_gate"));
        assertTrue(policy(DirectForwardMetalLinearOptions.defaults()
                .withNativeBf16Bf16ToF16(true, false))
                .allowNativeBf16Bf16ToF16Linear(gemma4, "logits"));
        assertFalse(policy(DirectForwardMetalLinearOptions.defaults()
                .withNativeBf16Bf16ToF16(true, true))
                .allowNativeBf16Bf16ToF16Linear(gemma4, "ffn_gate"));
    }

    @Test
    void nativeBf16ToF16ProjectionSpecificOverridesStayScoped() {
        ModelConfigTraits gemma4 = traits(true);

        assertTrue(policy(DirectForwardMetalLinearOptions.defaults()
                .withNativeBf16Bf16ToF16Ffn(true, false))
                .allowNativeBf16Bf16ToF16Linear(gemma4, "ffn_down"));
        assertFalse(policy(DirectForwardMetalLinearOptions.defaults()
                .withNativeBf16Bf16ToF16Ffn(true, false))
                .allowNativeBf16Bf16ToF16Linear(gemma4, "logits"));
        assertTrue(policy(DirectForwardMetalLinearOptions.defaults()
                .withNativeBf16Bf16ToF16Logits(true, false))
                .allowNativeBf16Bf16ToF16Linear(gemma4, "logits"));
        assertFalse(policy(DirectForwardMetalLinearOptions.defaults()
                .withNativeBf16Bf16ToF16Logits(true, true))
                .allowNativeBf16Bf16ToF16Linear(gemma4, "logits"));
    }

    @Test
    void nativeBf16ToF16RowsGateDecodeLogitsOnly() {
        ModelConfigTraits gemma4 = traits(true);
        DirectForwardMetalLinearRoutingPolicy policy = policy(
                DirectForwardMetalLinearOptions.defaults().withNativeBf16Bf16ToF16Logits(true, false));

        assertFalse(policy.allowNativeBf16Bf16ToF16LinearForRows(2, gemma4, "logits", true));
        assertFalse(policy.allowNativeBf16Bf16ToF16LinearForRows(1, gemma4, "logits", false));
        assertTrue(policy.allowNativeBf16Bf16ToF16LinearForRows(1, gemma4, "logits", true));
    }

    @Test
    void nonNativeBf16NeverUsesNativeBf16Bf16ToF16Path() {
        ModelConfigTraits generic = traits(false);

        assertFalse(policy(DirectForwardMetalLinearOptions.defaults()
                .withNativeBf16Bf16ToF16(true, false))
                .allowNativeBf16Bf16ToF16Linear(generic, "ffn_gate"));
    }

    private static ModelConfigTraits traits(boolean gemma4Text) {
        return new ModelConfigTraits(null, gemma4Text ? "gemma4_text" : "llama",
                0, 0, gemma4Text, false, false, false, false, false, false);
    }

    private static DirectForwardMetalLinearRoutingPolicy policy(DirectForwardMetalLinearOptions options) {
        return DirectForwardMetalLinearRoutingPolicy.from(options);
    }
}
