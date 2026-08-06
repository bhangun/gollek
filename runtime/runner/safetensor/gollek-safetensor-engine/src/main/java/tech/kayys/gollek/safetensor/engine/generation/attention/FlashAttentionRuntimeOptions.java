/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

final class FlashAttentionRuntimeOptions {
    private static final String EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.experimental_metal_bf16_linear";
    private static final String DISABLE_EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.disable_experimental_metal_bf16_linear";
    private static final String ENABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.enable_gemma4_metal_bf16_linear";
    private static final String DISABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.disable_gemma4_metal_bf16_linear";
    private static final String METAL_F16_WEIGHT_CACHE_MAX_BYTES_PROPERTY =
            "gollek.safetensor.metal_f16_weight_cache_max_bytes";
    private static final long DEFAULT_METAL_F16_WEIGHT_CACHE_MAX_BYTES = 2L * 1024L * 1024L * 1024L;
    private static final long METAL_F16_WEIGHT_CACHE_MAX_BYTES = Long.getLong(
            METAL_F16_WEIGHT_CACHE_MAX_BYTES_PROPERTY,
            DEFAULT_METAL_F16_WEIGHT_CACHE_MAX_BYTES);

    private static final String ENABLE_METAL_MIXED_HALF_LINEAR_PAIR_PROPERTY =
            "gollek.safetensor.enable_metal_mixed_half_linear_pair";
    private static final String DISABLE_METAL_MIXED_HALF_LINEAR_PAIR_PROPERTY =
            "gollek.safetensor.disable_metal_mixed_half_linear_pair";
    private static final String DISABLE_METAL_MIXED_HALF_LINEAR_TRIPLE_PROPERTY =
            "gollek.safetensor.disable_metal_mixed_half_linear_triple";

    private static final boolean METAL_MIXED_HALF_LINEAR_PAIR_ENABLED =
            resolveMetalMixedHalfLinearPairEnabled();

    private FlashAttentionRuntimeOptions() {
    }

    static long metalF16WeightCacheMaxBytes() {
        return METAL_F16_WEIGHT_CACHE_MAX_BYTES;
    }

    static boolean allowMetalBf16Linear() {
        if (Boolean.getBoolean(DISABLE_EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return true;
    }

    static boolean disableGemma4MetalBf16Linear() {
        return Boolean.getBoolean(DISABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY);
    }

    static String enableGemma4MetalBf16LinearValue() {
        return System.getProperty(ENABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY);
    }

    static boolean metalMixedHalfLinearPairEnabled() {
        return METAL_MIXED_HALF_LINEAR_PAIR_ENABLED;
    }

    static boolean disableMetalMixedHalfLinearTriple() {
        return Boolean.getBoolean(DISABLE_METAL_MIXED_HALF_LINEAR_TRIPLE_PROPERTY);
    }

    private static boolean resolveMetalMixedHalfLinearPairEnabled() {
        if (Boolean.getBoolean(DISABLE_METAL_MIXED_HALF_LINEAR_PAIR_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_METAL_MIXED_HALF_LINEAR_PAIR_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return true;
    }

}
