/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

record FlashAttentionRestrictedMetalOptions(
        boolean allowMetalRestrictedAttention,
        boolean disableMetalRestrictedAttention,
        boolean disableSlidingPrefillFa4Attention,
        boolean disableSharedDecodePackedAttention,
        String enableSharedDecodePackedAttentionValue,
        String enableSlidingPrefillFa4AttentionValue,
        boolean forceDenseRestrictedAttention) {
    private static final String LEGACY_ALLOW_METAL_RESTRICTED_ATTENTION_PROPERTY =
            "gollek.safetensor.allow_metal_gemma4_attention";
    private static final String LEGACY_DISABLE_METAL_RESTRICTED_ATTENTION_PROPERTY =
            "gollek.safetensor.disable_metal_gemma4_attention";
    private static final String LEGACY_ENABLE_SHARED_DECODE_PACKED_ATTENTION_PROPERTY =
            "gollek.safetensor.enable_gemma4_shared_decode_packed_attention";
    private static final String LEGACY_DISABLE_SHARED_DECODE_PACKED_ATTENTION_PROPERTY =
            "gollek.safetensor.disable_gemma4_shared_decode_packed_attention";
    private static final String LEGACY_ENABLE_SLIDING_PREFILL_FA4_ATTENTION_PROPERTY =
            "gollek.safetensor.enable_gemma4_sliding_prefill_fa4_attention";
    private static final String LEGACY_DISABLE_SLIDING_PREFILL_FA4_ATTENTION_PROPERTY =
            "gollek.safetensor.disable_gemma4_sliding_prefill_fa4_attention";
    private static final String LEGACY_FORCE_DENSE_RESTRICTED_ATTENTION_PROPERTY =
            "gollek.safetensor.force_dense_gemma4_attention";

    static FlashAttentionRestrictedMetalOptions fromSystemProperties() {
        return new FlashAttentionRestrictedMetalOptions(
                Boolean.getBoolean(LEGACY_ALLOW_METAL_RESTRICTED_ATTENTION_PROPERTY),
                Boolean.getBoolean(LEGACY_DISABLE_METAL_RESTRICTED_ATTENTION_PROPERTY),
                Boolean.getBoolean(LEGACY_DISABLE_SLIDING_PREFILL_FA4_ATTENTION_PROPERTY),
                Boolean.getBoolean(LEGACY_DISABLE_SHARED_DECODE_PACKED_ATTENTION_PROPERTY),
                System.getProperty(LEGACY_ENABLE_SHARED_DECODE_PACKED_ATTENTION_PROPERTY),
                System.getProperty(LEGACY_ENABLE_SLIDING_PREFILL_FA4_ATTENTION_PROPERTY),
                Boolean.getBoolean(LEGACY_FORCE_DENSE_RESTRICTED_ATTENTION_PROPERTY));
    }

    static FlashAttentionRestrictedMetalOptions defaults() {
        return new FlashAttentionRestrictedMetalOptions(false, false, false, false, null, null, false);
    }
}
