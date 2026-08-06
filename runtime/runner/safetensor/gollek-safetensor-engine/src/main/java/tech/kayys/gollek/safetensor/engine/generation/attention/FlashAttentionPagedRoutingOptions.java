/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

record FlashAttentionPagedRoutingOptions(
        boolean enableRestrictedPagedDecodeAttention,
        boolean disableRestrictedPagedDecodeAttention,
        boolean enableRawPagedSlidingDecodeAttention,
        boolean forceDecodeAttentionKernel,
        boolean disableDecodeAttentionKernel,
        int decodeAttentionGpuMinContext,
        int preferPagedMetalAttentionMaxTokens,
        String preferPagedMetalPrefillMaxTokensValue) {
    private static final String LEGACY_ENABLE_RESTRICTED_PAGED_DECODE_ATTENTION_PROPERTY =
            "gollek.safetensor.enable_gemma4_paged_decode_attention";
    private static final String LEGACY_DISABLE_RESTRICTED_PAGED_DECODE_ATTENTION_PROPERTY =
            "gollek.safetensor.disable_gemma4_paged_decode_attention";
    private static final String ENABLE_RAW_PAGED_SLIDING_DECODE_ATTENTION_PROPERTY =
            "gollek.safetensor.enable_raw_paged_sliding_decode_attention";
    private static final String PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS_PROPERTY =
            "gollek.safetensor.prefer_paged_metal_attention_max_tokens";
    private static final String PREFER_PAGED_METAL_PREFILL_MAX_TOKENS_PROPERTY =
            "gollek.safetensor.prefer_paged_metal_prefill_max_tokens";
    private static final String DECODE_ATTENTION_GPU_MIN_CONTEXT_ENV =
            "GOLLEK_METAL_DECODE_ATTENTION_GPU_MIN_CONTEXT";
    private static final String DISABLE_DECODE_ATTENTION_KERNEL_ENV =
            "GOLLEK_METAL_DISABLE_DECODE_ATTENTION_KERNEL";
    private static final String FORCE_DECODE_ATTENTION_KERNEL_ENV =
            "GOLLEK_METAL_FORCE_DECODE_ATTENTION_KERNEL";
    private static final int DEFAULT_DECODE_ATTENTION_GPU_MIN_CONTEXT = 64;
    private static final int DEFAULT_PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS = 1024;

    static FlashAttentionPagedRoutingOptions fromSystemPropertiesAndEnvironment() {
        return new FlashAttentionPagedRoutingOptions(
                Boolean.getBoolean(LEGACY_ENABLE_RESTRICTED_PAGED_DECODE_ATTENTION_PROPERTY),
                Boolean.getBoolean(LEGACY_DISABLE_RESTRICTED_PAGED_DECODE_ATTENTION_PROPERTY),
                Boolean.getBoolean(ENABLE_RAW_PAGED_SLIDING_DECODE_ATTENTION_PROPERTY),
                envTruthy(FORCE_DECODE_ATTENTION_KERNEL_ENV),
                envTruthy(DISABLE_DECODE_ATTENTION_KERNEL_ENV),
                Math.max(1, envIntOrDefault(DECODE_ATTENTION_GPU_MIN_CONTEXT_ENV,
                        DEFAULT_DECODE_ATTENTION_GPU_MIN_CONTEXT)),
                Integer.getInteger(PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS_PROPERTY,
                        DEFAULT_PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS),
                System.getProperty(PREFER_PAGED_METAL_PREFILL_MAX_TOKENS_PROPERTY));
    }

    static FlashAttentionPagedRoutingOptions defaults() {
        return new FlashAttentionPagedRoutingOptions(false, false, false, false, false,
                DEFAULT_DECODE_ATTENTION_GPU_MIN_CONTEXT, DEFAULT_PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS, null);
    }

    private static int envIntOrDefault(String name, int fallback) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            return fallback;
        }
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private static boolean envTruthy(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            return false;
        }
        String normalized = value.trim().toLowerCase();
        return !"0".equals(normalized)
                && !"false".equals(normalized)
                && !"no".equals(normalized)
                && !"off".equals(normalized);
    }
}
