/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionLinearOptionsTest {
    private static final String EXPERIMENTAL_METAL_LINEAR_PROPERTY =
            "gollek.safetensor.experimental_metal_linear";
    private static final String DISABLE_EXPERIMENTAL_METAL_LINEAR_PROPERTY =
            "gollek.safetensor.disable_experimental_metal_linear";

    @Test
    void defaultsEnableExperimentalMetalLinear() {
        assertTrue(FlashAttentionLinearOptions.defaults().experimentalMetalLinearEnabled());
    }

    @Test
    void explicitPropertyControlsExperimentalMetalLinear() {
        String previous = System.getProperty(EXPERIMENTAL_METAL_LINEAR_PROPERTY);
        try {
            System.setProperty(EXPERIMENTAL_METAL_LINEAR_PROPERTY, "false");

            assertFalse(FlashAttentionLinearOptions.fromSystemProperties().experimentalMetalLinearEnabled());
        } finally {
            restoreProperty(EXPERIMENTAL_METAL_LINEAR_PROPERTY, previous);
        }
    }

    @Test
    void disablePropertyOverridesExplicitEnable() {
        String previousExplicit = System.getProperty(EXPERIMENTAL_METAL_LINEAR_PROPERTY);
        String previousDisable = System.getProperty(DISABLE_EXPERIMENTAL_METAL_LINEAR_PROPERTY);
        try {
            System.setProperty(EXPERIMENTAL_METAL_LINEAR_PROPERTY, "true");
            System.setProperty(DISABLE_EXPERIMENTAL_METAL_LINEAR_PROPERTY, "true");

            assertFalse(FlashAttentionLinearOptions.fromSystemProperties().experimentalMetalLinearEnabled());
        } finally {
            restoreProperty(EXPERIMENTAL_METAL_LINEAR_PROPERTY, previousExplicit);
            restoreProperty(DISABLE_EXPERIMENTAL_METAL_LINEAR_PROPERTY, previousDisable);
        }
    }

    private static void restoreProperty(String name, String previous) {
        if (previous == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, previous);
        }
    }
}
