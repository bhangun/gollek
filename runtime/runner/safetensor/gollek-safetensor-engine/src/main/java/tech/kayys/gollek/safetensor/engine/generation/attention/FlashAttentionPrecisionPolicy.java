/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.metal.binding.MetalFlashAttentionBinding;

import java.util.Objects;

record FlashAttentionPrecisionPolicy(FlashAttentionPrecisionOptions options) {

    FlashAttentionPrecisionPolicy {
        options = Objects.requireNonNull(options, "options");
    }

    static FlashAttentionPrecisionPolicy from(FlashAttentionPrecisionOptions options) {
        return new FlashAttentionPrecisionPolicy(options);
    }

    boolean useBf16Attention(MetalFlashAttentionBinding fa4) {
        return options.useBf16Attention() && fa4 != null && fa4.isBf16Available();
    }
}
