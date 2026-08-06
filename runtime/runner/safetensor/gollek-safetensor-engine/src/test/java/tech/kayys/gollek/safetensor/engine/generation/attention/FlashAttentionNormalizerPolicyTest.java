/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelAttentionTraitsPolicy;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies policy decisions and reusable-buffer safety for attention
 * normalization routes.
 */
class FlashAttentionNormalizerPolicyTest {
    private static final String ENABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY =
            "gollek.safetensor.enable_metal_per_head_rms_norm";
    private static final String DISABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY =
            "gollek.safetensor.disable_metal_per_head_rms_norm";

    @Test
    void defaultsFollowModelPreference() {
        FlashAttentionNormalizerPolicy policy =
                FlashAttentionNormalizerPolicy.from(FlashAttentionNormalizerOptions.defaults());

        assertTrue(policy.shouldUseMetalPerHeadRmsNorm(preferMetalPerHeadRmsNormPolicy()));
        assertFalse(policy.shouldUseMetalPerHeadRmsNorm(genericPolicy()));
    }

    @Test
    void explicitEnableOverridesModelPreference() {
        FlashAttentionNormalizerPolicy enabled =
                FlashAttentionNormalizerPolicy.from(new FlashAttentionNormalizerOptions(false, "true"));
        FlashAttentionNormalizerPolicy disabled =
                FlashAttentionNormalizerPolicy.from(new FlashAttentionNormalizerOptions(false, "false"));

        assertTrue(enabled.shouldUseMetalPerHeadRmsNorm(genericPolicy()));
        assertFalse(disabled.shouldUseMetalPerHeadRmsNorm(preferMetalPerHeadRmsNormPolicy()));
    }

    @Test
    void disableWinsOverExplicitEnableAndModelPreference() {
        FlashAttentionNormalizerPolicy policy =
                FlashAttentionNormalizerPolicy.from(new FlashAttentionNormalizerOptions(true, "true"));

        assertFalse(policy.shouldUseMetalPerHeadRmsNorm(preferMetalPerHeadRmsNormPolicy()));
    }

    @Test
    void nullOptionsUseDefaults() {
        FlashAttentionNormalizerPolicy policy = FlashAttentionNormalizerPolicy.from(null);

        assertFalse(policy.shouldUseMetalPerHeadRmsNorm(genericPolicy()));
    }

    @Test
    void systemPropertiesControlRawNormalizerRequests() {
        String previousEnable = System.getProperty(ENABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY);
        String previousDisable = System.getProperty(DISABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY);
        try {
            System.setProperty(ENABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY, "true");
            System.setProperty(DISABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY, "true");

            FlashAttentionNormalizerOptions options = FlashAttentionNormalizerOptions.fromSystemProperties();

            assertTrue(options.disableMetalPerHeadRmsNorm());
            assertEquals("true", options.enableMetalPerHeadRmsNormValue());
        } finally {
            restoreProperty(ENABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY, previousEnable);
            restoreProperty(DISABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY, previousDisable);
        }
    }

    @Test
    void reusableInputNormalizersBypassMetalEvenWhenModelPrefersMetal() {
        FlashAttentionNormalizer normalizer = new FlashAttentionNormalizer(() -> {
            throw new AssertionError("Reusable in-place normalization must not request Metal binding");
        }, new FlashAttentionNormalizerOptions(false, "true"));
        FlashAttentionModelPolicy modelPolicy = preferMetalPerHeadRmsNormPolicy();

        try (AccelTensor x = AccelTensor.fromFloatArray(new float[] {
                1.0f, 2.0f, 3.0f, 4.0f
        }, 1, 1, 4);
                AccelTensor weight = AccelTensor.ones(4)) {
            assertSame(x, normalizer.rmsNormReusingInput(x, weight, 1e-6, false));
            assertSame(x, normalizer.perHeadRmsNormReusingInput(x, weight, 1e-6, false, modelPolicy));
            assertSame(x, normalizer.perHeadRmsNormNoWeightReusingInput(x, 1e-6, modelPolicy));
        }
    }

    private static FlashAttentionModelPolicy preferMetalPerHeadRmsNormPolicy() {
        ModelRuntimeTraits traits = ModelRuntimeTraits.builder()
                .nativeBf16Matvec()
                .attention(ModelAttentionTraitsPolicy.nativeBf16Matvec())
                .build();
        return FlashAttentionModelPolicy.resolve(null, new ModelConfig(), traits);
    }

    private static FlashAttentionModelPolicy genericPolicy() {
        return FlashAttentionModelPolicy.resolve(null, new ModelConfig(), ModelRuntimeTraits.EMPTY);
    }

    private static void restoreProperty(String name, String previous) {
        if (previous == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, previous);
        }
    }
}
