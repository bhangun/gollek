/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardFfnFastPathRoutingPolicyTest {

    @Test
    void gemma4UsesMetalGegluFusedFfnByDefault() {
        assertTrue(policy(DirectForwardFfnFastPathOptions.defaults()).shouldUseMetalGegluFusedFfn(gemma4()));
        assertFalse(policy(DirectForwardFfnFastPathOptions.defaults()).shouldUseMetalGegluFusedFfn(generic()));
        assertTrue(policy(DirectForwardFfnFastPathOptions.defaults()
                .withMetalFusedFfn(false, false, true))
                .shouldUseMetalGegluFusedFfn(generic()));
    }

    @Test
    void localFusedHalfFfnSkipsNativeBf16UnlessExplicitlyAllowed() {
        assertTrue(policy(DirectForwardFfnFastPathOptions.defaults()).shouldTryLocalFusedHalfFfn(generic()));
        assertFalse(policy(DirectForwardFfnFastPathOptions.defaults()).shouldTryLocalFusedHalfFfn(gemma4()));
        assertTrue(policy(DirectForwardFfnFastPathOptions.defaults()
                .withNativeBf16FusedHalfFfn(true, false))
                .shouldTryLocalFusedHalfFfn(gemma4()));
    }

    @Test
    void nativeBf16FusedHalfFfnGlobalFlagHonorsDisableAndExplicit() {
        assertFalse(policy(DirectForwardFfnFastPathOptions.defaults()).isNativeBf16FusedHalfFfnAllowed());
        assertTrue(policy(DirectForwardFfnFastPathOptions.defaults()
                .withNativeBf16FusedHalfFfn(true, false))
                .isNativeBf16FusedHalfFfnAllowed());
        assertFalse(policy(DirectForwardFfnFastPathOptions.defaults()
                .withNativeBf16FusedHalfFfn(true, true))
                .isNativeBf16FusedHalfFfnAllowed());
    }

    @Test
    void nativeBf16FusedHalfFfnPrefillRequiresEffectiveMinimumRows() {
        DirectForwardFfnFastPathRoutingPolicy policy = policy(
                DirectForwardFfnFastPathOptions.defaults().withMetalFusedFfnPrefill(null, 3));

        assertFalse(policy.allowNativeBf16FusedHalfFfn(1, gemma4()));
        assertFalse(policy.allowNativeBf16FusedHalfFfn(2, gemma4()));
        assertTrue(policy.allowNativeBf16FusedHalfFfn(3, gemma4()));
        assertTrue(policy.allowNativeBf16FusedHalfFfn(1, generic()));
    }

    @Test
    void nativeBf16FusedHalfFfnPrefillExplicitStillRejectsDecodeRows() {
        DirectForwardFfnFastPathRoutingPolicy policy = policy(
                DirectForwardFfnFastPathOptions.defaults().withMetalFusedFfnPrefill(true, 2));

        assertFalse(policy.allowNativeBf16FusedHalfFfn(1, gemma4()));
        assertTrue(policy.allowNativeBf16FusedHalfFfn(2, gemma4()));
        assertFalse(policy(DirectForwardFfnFastPathOptions.defaults()
                .withMetalFusedFfnPrefill(false, 2))
                .allowNativeBf16FusedHalfFfn(2, gemma4()));
    }

    @Test
    void matvecFfnDefaultsFollowModelFamiliesAndDisableWins() {
        DirectForwardFfnFastPathRoutingPolicy defaults = policy(DirectForwardFfnFastPathOptions.defaults());

        assertTrue(defaults.shouldUseMetalGegluMatvecFfn(gemma3()));
        assertTrue(defaults.shouldUseMetalGegluMatvecFfn(gemma4()));
        assertFalse(defaults.shouldUseMetalGegluMatvecFfn(generic()));
        assertTrue(defaults.shouldUseMetalSwigluMatvecFfn(qwen()));
        assertFalse(defaults.shouldUseMetalSwigluMatvecFfn(generic()));
        assertFalse(defaults.shouldUseMetalGateUpMatvecFfn());

        DirectForwardFfnFastPathRoutingPolicy disabled = policy(
                DirectForwardFfnFastPathOptions.defaults().withMetalMatvecFfn(true, true, true, true));
        assertFalse(disabled.shouldUseMetalGegluMatvecFfn(gemma4()));
        assertFalse(disabled.shouldUseMetalSwigluMatvecFfn(qwen()));
        assertFalse(disabled.shouldUseMetalGateUpMatvecFfn());
    }

    @Test
    void matvecFfnExplicitOverridesFamilyDefaults() {
        DirectForwardFfnFastPathRoutingPolicy options = policy(
                DirectForwardFfnFastPathOptions.defaults().withMetalMatvecFfn(true, true, true, false));

        assertTrue(options.shouldUseMetalGegluMatvecFfn(generic()));
        assertTrue(options.shouldUseMetalSwigluMatvecFfn(generic()));
        assertTrue(options.shouldUseMetalGateUpMatvecFfn());

        DirectForwardFfnFastPathRoutingPolicy rejected = policy(
                DirectForwardFfnFastPathOptions.defaults().withMetalMatvecFfn(false, false, false, false));
        assertFalse(rejected.shouldUseMetalGegluMatvecFfn(gemma4()));
        assertFalse(rejected.shouldUseMetalSwigluMatvecFfn(qwen()));
        assertFalse(rejected.shouldUseMetalGateUpMatvecFfn());
    }

    @Test
    void rowPrefillMatvecFfnIsNativeBf16OnlyOptInAndBounded() {
        DirectForwardFfnFastPathRoutingPolicy defaults = policy(DirectForwardFfnFastPathOptions.defaults());
        DirectForwardFfnFastPathRoutingPolicy enabled = policy(
                DirectForwardFfnFastPathOptions.defaults().withMetalMatvecFfnPrefillRows(true, 4));
        DirectForwardFfnFastPathRoutingPolicy disabled = policy(
                DirectForwardFfnFastPathOptions.defaults().withMetalMatvecFfnPrefillRows(true, 4)
                        .withMetalMatvecFfn(true, true, true, true));

        assertFalse(defaults.shouldUseMetalMatvecFfnPrefillRows(gemma4(), 2));
        assertFalse(enabled.shouldUseMetalMatvecFfnPrefillRows(gemma4(), 1));
        assertTrue(enabled.shouldUseMetalMatvecFfnPrefillRows(gemma4(), 4));
        assertFalse(enabled.shouldUseMetalMatvecFfnPrefillRows(gemma4(), 5));
        assertFalse(enabled.shouldUseMetalMatvecFfnPrefillRows(generic(), 2));
        assertFalse(disabled.shouldUseMetalMatvecFfnPrefillRows(gemma4(), 2));
    }

    @Test
    void gemma4PrefillPrefersFusedFfnBeforeExperimentalRowMatvec() {
        DirectForwardFfnFastPathRoutingPolicy defaults = policy(DirectForwardFfnFastPathOptions.defaults());
        DirectForwardFfnFastPathRoutingPolicy rowPrefillEnabled = policy(
                DirectForwardFfnFastPathOptions.defaults().withMetalMatvecFfnPrefillRows(true, 16));
        DirectForwardFfnFastPathRoutingPolicy rowPrefillPreferred = policy(
                DirectForwardFfnFastPathOptions.defaults().withMetalMatvecFfnPrefillRows(true, 16, true));

        assertTrue(defaults.shouldPreferMetalFusedFfnPrefillOverMatvecRows(gemma4(), 12));
        assertTrue(rowPrefillEnabled.shouldPreferMetalFusedFfnPrefillOverMatvecRows(gemma4(), 12));
        assertFalse(rowPrefillPreferred.shouldPreferMetalFusedFfnPrefillOverMatvecRows(gemma4(), 12));
        assertFalse(rowPrefillEnabled.shouldPreferMetalFusedFfnPrefillOverMatvecRows(gemma4(), 1));
        assertFalse(rowPrefillEnabled.shouldPreferMetalFusedFfnPrefillOverMatvecRows(generic(), 12));
    }

    @Test
    void runtimeRowPrefillPreferenceOverridesFusedFfnStrategy() {
        String previous = System.getProperty(
                DirectForwardFfnFastPathOptions.PREFER_METAL_MATVEC_FFN_PREFILL_ROWS_PROPERTY);
        try {
            DirectForwardFfnFastPathRoutingPolicy defaults = policy(DirectForwardFfnFastPathOptions.defaults());

            System.clearProperty(DirectForwardFfnFastPathOptions.PREFER_METAL_MATVEC_FFN_PREFILL_ROWS_PROPERTY);
            assertTrue(defaults.shouldPreferMetalFusedFfnPrefillOverMatvecRows(gemma4(), 12));

            System.setProperty(DirectForwardFfnFastPathOptions.PREFER_METAL_MATVEC_FFN_PREFILL_ROWS_PROPERTY, "true");
            assertFalse(defaults.shouldPreferMetalFusedFfnPrefillOverMatvecRows(gemma4(), 12));
        } finally {
            if (previous == null) {
                System.clearProperty(DirectForwardFfnFastPathOptions.PREFER_METAL_MATVEC_FFN_PREFILL_ROWS_PROPERTY);
            } else {
                System.setProperty(DirectForwardFfnFastPathOptions.PREFER_METAL_MATVEC_FFN_PREFILL_ROWS_PROPERTY,
                        previous);
            }
        }
    }

    @Test
    void fusedFfnPreferenceHonorsExplicitDisables() {
        DirectForwardFfnFastPathRoutingPolicy fusedPrefillDisabled = policy(
                DirectForwardFfnFastPathOptions.defaults().withMetalFusedFfnPrefill(false, 2));
        DirectForwardFfnFastPathRoutingPolicy fusedDisabled = policy(
                DirectForwardFfnFastPathOptions.defaults().withMetalFusedFfn(true, false, true));

        assertFalse(fusedPrefillDisabled.shouldPreferMetalFusedFfnPrefillOverMatvecRows(gemma4(), 12));
        assertFalse(fusedDisabled.shouldPreferMetalFusedFfnPrefillOverMatvecRows(gemma4(), 12));
    }

    @Test
    void validationCanBeForcedOrEnabledByTrace() {
        assertFalse(policy(DirectForwardFfnFastPathOptions.defaults()).shouldValidateMetalMatvecFfn(false));
        assertTrue(policy(DirectForwardFfnFastPathOptions.defaults()).shouldValidateMetalMatvecFfn(true));
        assertTrue(policy(DirectForwardFfnFastPathOptions.defaults()
                .withValidateMetalMatvecFfn(true))
                .shouldValidateMetalMatvecFfn(false));
    }

    private static DirectForwardFfnFastPathRoutingPolicy policy(DirectForwardFfnFastPathOptions options) {
        return DirectForwardFfnFastPathRoutingPolicy.from(options);
    }

    private static ModelConfigTraits generic() {
        return traits("llama", false, false, false, false);
    }

    private static ModelConfigTraits gemma3() {
        return traits("gemma3_text", false, true, false, false);
    }

    private static ModelConfigTraits gemma4() {
        return traits("gemma4_text", true, false, false, false);
    }

    private static ModelConfigTraits qwen() {
        return traits("qwen2", false, false, true, false);
    }

    private static ModelConfigTraits traits(
            String modelType,
            boolean gemma4Text,
            boolean gemma3Text,
            boolean qwenText,
            boolean gemma4StylePerLayerInputs) {
        return new ModelConfigTraits(
                null, modelType, 0, 0, gemma4Text, gemma3Text, qwenText, gemma4StylePerLayerInputs, false, false, false);
    }
}
