/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

record FlashAttentionFa4RoutingOptions(boolean disableFa4Attention) {
    private static final String DISABLE_FA4_ATTENTION_PROPERTY =
            "gollek.safetensor.disable_fa4_attention";

    static FlashAttentionFa4RoutingOptions fromSystemProperties() {
        return new FlashAttentionFa4RoutingOptions(Boolean.getBoolean(DISABLE_FA4_ATTENTION_PROPERTY));
    }

    static FlashAttentionFa4RoutingOptions defaults() {
        return new FlashAttentionFa4RoutingOptions(false);
    }
}
