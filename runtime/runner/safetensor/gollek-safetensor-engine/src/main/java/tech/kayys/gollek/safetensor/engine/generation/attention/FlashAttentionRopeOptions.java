/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

record FlashAttentionRopeOptions(
        boolean legacyInterleavedGemma4Rope,
        boolean experimentalGemma4SplitHalfRope) {
    private static final String EXPERIMENTAL_GEMMA4_SPLIT_HALF_ROPE_PROPERTY =
            "gollek.safetensor.experimental_gemma4_split_half_rope";
    private static final String LEGACY_INTERLEAVED_GEMMA4_ROPE_PROPERTY =
            "gollek.safetensor.legacy_interleaved_gemma4_rope";

    static FlashAttentionRopeOptions fromSystemProperties() {
        return new FlashAttentionRopeOptions(
                Boolean.getBoolean(LEGACY_INTERLEAVED_GEMMA4_ROPE_PROPERTY),
                Boolean.getBoolean(EXPERIMENTAL_GEMMA4_SPLIT_HALF_ROPE_PROPERTY));
    }

    static FlashAttentionRopeOptions defaults() {
        return new FlashAttentionRopeOptions(false, false);
    }
}
