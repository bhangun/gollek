/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

record FlashAttentionNormalizerOptions(
        boolean disableMetalPerHeadRmsNorm,
        String enableMetalPerHeadRmsNormValue) {
    private static final String ENABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY =
            "gollek.safetensor.enable_metal_per_head_rms_norm";
    private static final String DISABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY =
            "gollek.safetensor.disable_metal_per_head_rms_norm";

    static FlashAttentionNormalizerOptions fromSystemProperties() {
        return new FlashAttentionNormalizerOptions(
                Boolean.getBoolean(DISABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY),
                System.getProperty(ENABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY));
    }

    static FlashAttentionNormalizerOptions defaults() {
        return new FlashAttentionNormalizerOptions(false, null);
    }
}
