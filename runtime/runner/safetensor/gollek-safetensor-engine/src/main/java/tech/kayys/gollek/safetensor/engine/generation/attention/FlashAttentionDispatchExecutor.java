/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import java.util.Objects;

final class FlashAttentionDispatchExecutor {
    private final FlashAttentionMetalDispatchBackend metalBackend;
    private final FlashAttentionJavaDispatchBackend javaBackend;

    FlashAttentionDispatchExecutor(FlashAttentionMetalAttention metalAttention) {
        this(metalAttention, PagedAttentionVectorOptions.defaults());
    }

    FlashAttentionDispatchExecutor(FlashAttentionMetalAttention metalAttention,
            PagedAttentionVectorOptions pagedAttentionOptions) {
        this(new FlashAttentionMetalDispatchAdapter(metalAttention),
                new FlashAttentionJavaDispatchAdapter(pagedAttentionOptions));
    }

    FlashAttentionDispatchExecutor(FlashAttentionMetalDispatchBackend metalBackend,
            FlashAttentionJavaDispatchBackend javaBackend) {
        this.metalBackend = Objects.requireNonNull(metalBackend, "metalBackend");
        this.javaBackend = Objects.requireNonNull(javaBackend, "javaBackend");
    }

    AccelTensor execute(FlashAttentionDispatchPath path, FlashAttentionDispatchRequest request) {
        return switch (path) {
            case DENSE_SHARED -> denseSharedAttention(request);
            case DENSE_RESTRICTED_JAVA -> denseRestrictedJavaAttention(request);
            case FA4_PAGED_METAL, METAL_TILED -> tiledMetalAttention(request);
            case SLIDING_DECODE_METAL -> slidingDecodeMetalAttention(request);
            case PAGED_JAVA -> pagedJavaAttention(request);
        };
    }

    private AccelTensor denseSharedAttention(FlashAttentionDispatchRequest request) {
        AccelTensor metalSharedOut = metalBackend.denseSharedAttention(request);
        if (metalSharedOut != null) {
            return metalSharedOut;
        }
        return javaBackend.denseSharedAttention(request);
    }

    private AccelTensor denseRestrictedJavaAttention(FlashAttentionDispatchRequest request) {
        return javaBackend.denseRestrictedAttention(request);
    }

    private AccelTensor tiledMetalAttention(FlashAttentionDispatchRequest request) {
        return metalBackend.tiledAttention(request);
    }

    private AccelTensor slidingDecodeMetalAttention(FlashAttentionDispatchRequest request) {
        return metalBackend.slidingDecodeAttention(request);
    }

    private AccelTensor pagedJavaAttention(FlashAttentionDispatchRequest request) {
        return javaBackend.pagedAttention(request);
    }
}
