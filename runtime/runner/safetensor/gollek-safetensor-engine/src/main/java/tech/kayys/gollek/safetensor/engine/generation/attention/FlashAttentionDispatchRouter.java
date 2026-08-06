/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

final class FlashAttentionDispatchRouter {
    private final FlashAttentionRoutingPolicy routing;

    FlashAttentionDispatchRouter(FlashAttentionRoutingPolicy routing) {
        this.routing = routing;
    }

    FlashAttentionDispatchPath select(FlashAttentionDispatchRequest request) {
        if (request.useDenseSharedKvState()) {
            return FlashAttentionDispatchPath.DENSE_SHARED;
        }
        if (routing.canUseDenseRestrictedAttention(request.config(), request.modelPolicy(), request.layerIdx())
                && !request.kvSession().isQuantized()) {
            return FlashAttentionDispatchPath.DENSE_RESTRICTED_JAVA;
        }
        if (routing.canUseFa4PagedAttention(request.config(), request.layerIdx(), request.attentionSoftCap())) {
            return FlashAttentionDispatchPath.FA4_PAGED_METAL;
        }
        if (routing.canUseSlidingDecodeMetalAttention(request.config(), request.modelPolicy(), request.layerIdx(),
                request.seqLen())) {
            return FlashAttentionDispatchPath.SLIDING_DECODE_METAL;
        }
        if (routing.canUseMetalAttention(request.config(), request.modelPolicy(), request.layerIdx(),
                request.seqLen(), request.startPos(), request.attentionSoftCap())) {
            return FlashAttentionDispatchPath.METAL_TILED;
        }
        return FlashAttentionDispatchPath.PAGED_JAVA;
    }
}
