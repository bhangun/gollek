/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import java.util.Objects;

final class FlashAttentionMetalDispatchAdapter implements FlashAttentionMetalDispatchBackend {
    private final FlashAttentionMetalAttention metalAttention;

    FlashAttentionMetalDispatchAdapter(FlashAttentionMetalAttention metalAttention) {
        this.metalAttention = Objects.requireNonNull(metalAttention, "metalAttention");
    }

    @Override
    public AccelTensor denseSharedAttention(FlashAttentionDispatchRequest request) {
        return metalAttention.denseSharedAttention(
                request.query(), request.key(), request.value(), request.sharedKvState(), request.config(),
                request.modelPolicy(), request.layerIdx(), request.startPos(), request.numQueryHeads(),
                request.numKeyValueHeads(), request.headDim(), request.scale(), request.causal(),
                request.attentionSoftCap(), request.attentionContextBuffer());
    }

    @Override
    public AccelTensor tiledAttention(FlashAttentionDispatchRequest request) {
        return metalAttention.tiledAttention(
                request.query(), request.kvSession(), request.kvLayerIdx(), request.startPos(),
                request.numQueryHeads(), request.numKeyValueHeads(), request.headDim(), request.scale(),
                request.causal(), request.attentionSoftCap(), request.config(), request.modelPolicy(),
                request.layerIdx(), request.attentionContextBuffer());
    }

    @Override
    public AccelTensor slidingDecodeAttention(FlashAttentionDispatchRequest request) {
        return metalAttention.slidingDecodeAttention(
                request.query(), request.kvSession(), request.layerIdx(), request.kvLayerIdx(),
                request.startPos(), request.numQueryHeads(), request.numKeyValueHeads(), request.headDim(),
                request.scale(), request.attentionSoftCap(), request.config(), request.modelPolicy(),
                request.attentionContextBuffer());
    }
}
