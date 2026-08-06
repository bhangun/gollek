/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionMatvecPolicyTest {
    private static final String ENABLE_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.enable_metal_half_matvec";
    private static final String METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_half_matvec_max_output";

    @Test
    void defaultsAllowCompactHalfMatvecWithinThreshold() {
        FlashAttentionMatvecPolicy policy = FlashAttentionMatvecPolicy.from(FlashAttentionMatvecOptions.defaults());
        FlashAttentionModelPolicy compactPolicy = policy(false, true, false);

        assertTrue(policy.shouldUseMetalHalfMatvec(compactPolicy, 8192));
        assertFalse(policy.shouldUseMetalHalfMatvec(compactPolicy, 8193));
    }

    @Test
    void disabledHalfMatvecOverridesModelCandidate() {
        FlashAttentionMatvecPolicy policy = FlashAttentionMatvecPolicy.from(new FlashAttentionMatvecOptions(
                true, null, null, null, 8192,
                false, null, false, 262144,
                false, 4096));

        assertFalse(policy.shouldUseMetalHalfMatvec(policy(true, true, true), 1));
    }

    @Test
    void autoHalfMatvecCanEnableLargeCandidate() {
        FlashAttentionMatvecPolicy policy = FlashAttentionMatvecPolicy.from(new FlashAttentionMatvecOptions(
                false, null, null, true, 8192,
                false, null, false, 262144,
                false, 4096));

        assertTrue(policy.shouldUseMetalHalfMatvec(policy(false, false, true), 8192));
    }

    @Test
    void transposedHalfMatvecRequiresExplicitEnableAndEligiblePolicy() {
        FlashAttentionMatvecPolicy policy = FlashAttentionMatvecPolicy.from(new FlashAttentionMatvecOptions(
                false, null, null, null, 8192,
                false, true, false, 4,
                false, 4096));

        assertTrue(policy.shouldUseMetalTransposedHalfMatvec(policy(true, false, false), 4));
        assertFalse(policy.shouldUseMetalTransposedHalfMatvec(policy(false, false, false), 4));
        assertFalse(policy.shouldUseMetalTransposedHalfMatvec(policy(true, false, false), 5));
    }

    @Test
    void transposedHalfMatvecCanAllowAllExperimentalProjectors() {
        FlashAttentionMatvecPolicy policy = FlashAttentionMatvecPolicy.from(new FlashAttentionMatvecOptions(
                false, null, null, null, 8192,
                false, true, true, 4,
                false, 4096));

        assertTrue(policy.shouldUseMetalTransposedHalfMatvec(policy(false, false, false), 4));
    }

    @Test
    void tripleMatvecRequiresPositiveDimsWithinThreshold() {
        FlashAttentionMatvecPolicy policy = FlashAttentionMatvecPolicy.from(new FlashAttentionMatvecOptions(
                false, null, null, null, 8192,
                false, null, false, 262144,
                false, 6));

        assertTrue(policy.shouldUseMetalHalfTripleMatvec(1, 2, 3));
        assertFalse(policy.shouldUseMetalHalfTripleMatvec(1, 2, 4));
        assertFalse(policy.shouldUseMetalHalfTripleMatvec(0, 2, 3));
    }

    @Test
    void nullOptionsUseDefaults() {
        FlashAttentionMatvecPolicy policy = FlashAttentionMatvecPolicy.from(null);

        assertFalse(policy.shouldUseMetalHalfMatvec(policy(false, true, false), 8193));
    }

    @Test
    void systemPropertiesControlRawMatvecRequests() {
        String previousEnable = System.getProperty(ENABLE_METAL_HALF_MATVEC_PROPERTY);
        String previousMaxOutput = System.getProperty(METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY);
        try {
            System.setProperty(ENABLE_METAL_HALF_MATVEC_PROPERTY, "true");
            System.setProperty(METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY, "123");

            FlashAttentionMatvecOptions options = FlashAttentionMatvecOptions.fromSystemProperties();

            assertEquals(Boolean.TRUE, options.enableMetalHalfMatvec());
            assertEquals(123, options.metalHalfMatvecMaxOutput());
        } finally {
            restoreProperty(ENABLE_METAL_HALF_MATVEC_PROPERTY, previousEnable);
            restoreProperty(METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY, previousMaxOutput);
        }
    }

    private static FlashAttentionModelPolicy policy(boolean preferNativeBf16, boolean compactCandidate,
            boolean largeCandidate) {
        ModelRuntimeTraits.AttentionRuntimeTraits attention =
                new ModelRuntimeTraits.AttentionRuntimeTraits(
                        false,
                        false,
                        false,
                        preferNativeBf16,
                        false,
                        false,
                        false,
                        0,
                        compactCandidate,
                        largeCandidate,
                        false);
        ModelRuntimeTraits traits = ModelRuntimeTraits.builder()
                .attention(attention)
                .build();
        return FlashAttentionModelPolicy.resolve(null, new ModelConfig(), traits);
    }

    private static void restoreProperty(String name, String previous) {
        if (previous == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, previous);
        }
    }
}
