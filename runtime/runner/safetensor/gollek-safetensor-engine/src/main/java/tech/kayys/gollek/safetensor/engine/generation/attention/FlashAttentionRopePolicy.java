/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

record FlashAttentionRopePolicy(FlashAttentionRopeOptions options) {

    FlashAttentionRopePolicy {
        if (options == null) {
            options = FlashAttentionRopeOptions.defaults();
        }
    }

    static FlashAttentionRopePolicy from(FlashAttentionRopeOptions options) {
        return new FlashAttentionRopePolicy(options);
    }

    boolean useInterleavedRope(FlashAttentionModelPolicy modelPolicy) {
        if (modelPolicy == null) {
            return false;
        }
        boolean legacyGemma4Interleaved =
                modelPolicy.nativeBf16Matvec() && options.legacyInterleavedGemma4Rope();
        return modelPolicy.useInterleavedRope(
                legacyGemma4Interleaved,
                options.experimentalGemma4SplitHalfRope());
    }
}
