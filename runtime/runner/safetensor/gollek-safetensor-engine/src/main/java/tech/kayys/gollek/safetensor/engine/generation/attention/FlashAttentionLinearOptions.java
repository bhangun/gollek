/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

record FlashAttentionLinearOptions(boolean experimentalMetalLinearEnabled) {
    private static final String EXPERIMENTAL_METAL_LINEAR_PROPERTY =
            "gollek.safetensor.experimental_metal_linear";
    private static final String DISABLE_EXPERIMENTAL_METAL_LINEAR_PROPERTY =
            "gollek.safetensor.disable_experimental_metal_linear";

    static FlashAttentionLinearOptions fromSystemProperties() {
        return new FlashAttentionLinearOptions(resolveExperimentalMetalLinearEnabled());
    }

    static FlashAttentionLinearOptions defaults() {
        return new FlashAttentionLinearOptions(true);
    }

    private static boolean resolveExperimentalMetalLinearEnabled() {
        if (Boolean.getBoolean(DISABLE_EXPERIMENTAL_METAL_LINEAR_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(EXPERIMENTAL_METAL_LINEAR_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return true;
    }
}
