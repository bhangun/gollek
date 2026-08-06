/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

record FlashAttentionNormalizationOptions(
        boolean disableGemma4QkNorm,
        boolean disableGemma4ValueNorm) {
    private static final String DISABLE_GEMMA4_V_NORM_PROPERTY =
            "gollek.safetensor.disable_gemma4_v_norm";
    private static final String DISABLE_GEMMA4_QK_NORM_PROPERTY =
            "gollek.safetensor.disable_gemma4_qk_norm";

    static FlashAttentionNormalizationOptions fromSystemProperties() {
        return new FlashAttentionNormalizationOptions(
                Boolean.getBoolean(DISABLE_GEMMA4_QK_NORM_PROPERTY),
                Boolean.getBoolean(DISABLE_GEMMA4_V_NORM_PROPERTY));
    }

    static FlashAttentionNormalizationOptions defaults() {
        return new FlashAttentionNormalizationOptions(false, false);
    }
}
