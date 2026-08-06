/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

interface FlashAttentionMetalDispatchBackend {
    AccelTensor denseSharedAttention(FlashAttentionDispatchRequest request);

    AccelTensor tiledAttention(FlashAttentionDispatchRequest request);

    AccelTensor slidingDecodeAttention(FlashAttentionDispatchRequest request);
}
