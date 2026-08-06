/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

record FlashAttentionPrecisionOptions(boolean useBf16Attention) {
    private static final String USE_BF16_ATTENTION_PROPERTY = "gollek.safetensor.use_bf16_attention";

    static FlashAttentionPrecisionOptions fromSystemProperties() {
        return new FlashAttentionPrecisionOptions(Boolean.getBoolean(USE_BF16_ATTENTION_PROPERTY));
    }

    static FlashAttentionPrecisionOptions defaults() {
        return new FlashAttentionPrecisionOptions(false);
    }
}
