/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

record FlashAttentionMatvecOptions(
        boolean disableMetalHalfMatvec,
        Boolean enableMetalHalfMatvec,
        Boolean autoMetalAttentionHalfMatvec,
        Boolean autoMetalHalfMatvec,
        int metalHalfMatvecMaxOutput,
        boolean disableMetalTransposedHalfMatvec,
        Boolean enableMetalTransposedHalfMatvec,
        boolean enableMetalTransposedHalfMatvecAll,
        int metalTransposedHalfMatvecMaxOutput,
        boolean disableMetalMixedHalfLinearTripleMatvec,
        int metalMixedHalfLinearTripleMatvecMaxOutput) {
    private static final String ENABLE_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.enable_metal_half_matvec";
    private static final String DISABLE_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_half_matvec";
    private static final String AUTO_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.auto_metal_half_matvec";
    private static final String AUTO_METAL_ATTENTION_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.auto_metal_attention_half_matvec";
    private static final String METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_half_matvec_max_output";
    private static final int DEFAULT_METAL_HALF_MATVEC_MAX_OUTPUT = 8192;
    private static final String ENABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.enable_metal_transposed_half_matvec";
    private static final String ENABLE_METAL_TRANSPOSED_HALF_MATVEC_ALL_PROPERTY =
            "gollek.safetensor.enable_metal_transposed_half_matvec_all";
    private static final String DISABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_transposed_half_matvec";
    private static final String METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_transposed_half_matvec_max_output";
    private static final int DEFAULT_METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT = 262144;
    private static final String DISABLE_METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_mixed_half_linear_triple_matvec";
    private static final String METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_mixed_half_linear_triple_matvec_max_output";
    private static final int DEFAULT_METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_MAX_OUTPUT = 8192;

    static FlashAttentionMatvecOptions fromSystemProperties() {
        return new FlashAttentionMatvecOptions(
                Boolean.getBoolean(DISABLE_METAL_HALF_MATVEC_PROPERTY),
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_HALF_MATVEC_PROPERTY)),
                parseOptionalBoolean(System.getProperty(AUTO_METAL_ATTENTION_HALF_MATVEC_PROPERTY)),
                parseOptionalBoolean(System.getProperty(AUTO_METAL_HALF_MATVEC_PROPERTY)),
                Integer.getInteger(METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY,
                        DEFAULT_METAL_HALF_MATVEC_MAX_OUTPUT),
                Boolean.getBoolean(DISABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY),
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY)),
                Boolean.getBoolean(ENABLE_METAL_TRANSPOSED_HALF_MATVEC_ALL_PROPERTY),
                Integer.getInteger(METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT_PROPERTY,
                        DEFAULT_METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT),
                Boolean.getBoolean(DISABLE_METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_PROPERTY),
                Integer.getInteger(METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_MAX_OUTPUT_PROPERTY,
                        DEFAULT_METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_MAX_OUTPUT));
    }

    static FlashAttentionMatvecOptions defaults() {
        return new FlashAttentionMatvecOptions(
                false,
                null,
                null,
                null,
                DEFAULT_METAL_HALF_MATVEC_MAX_OUTPUT,
                false,
                null,
                false,
                DEFAULT_METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT,
                false,
                DEFAULT_METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_MAX_OUTPUT);
    }

    private static Boolean parseOptionalBoolean(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return Boolean.parseBoolean(value);
    }
}
