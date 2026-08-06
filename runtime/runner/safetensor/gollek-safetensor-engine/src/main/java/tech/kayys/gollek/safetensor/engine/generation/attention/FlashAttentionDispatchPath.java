/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

enum FlashAttentionDispatchPath {
    DENSE_SHARED,
    DENSE_RESTRICTED_JAVA,
    FA4_PAGED_METAL,
    SLIDING_DECODE_METAL,
    METAL_TILED,
    PAGED_JAVA
}
