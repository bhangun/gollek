/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

interface FlashAttentionJavaDispatchBackend {
    AccelTensor denseSharedAttention(FlashAttentionDispatchRequest request);

    AccelTensor denseRestrictedAttention(FlashAttentionDispatchRequest request);

    AccelTensor pagedAttention(FlashAttentionDispatchRequest request);
}
