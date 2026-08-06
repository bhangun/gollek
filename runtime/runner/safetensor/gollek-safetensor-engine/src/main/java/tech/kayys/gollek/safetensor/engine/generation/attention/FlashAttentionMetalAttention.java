/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

final class FlashAttentionMetalAttention {
    private final FlashAttentionMetalDenseSharedAttention denseShared;
    private final FlashAttentionMetalTiledAttention tiled;
    private final FlashAttentionMetalSlidingDecodeAttention slidingDecode;

    FlashAttentionMetalAttention(FlashAttentionMetalDenseSharedAttention denseShared,
            FlashAttentionMetalTiledAttention tiled,
            FlashAttentionMetalSlidingDecodeAttention slidingDecode) {
        this.denseShared = Objects.requireNonNull(denseShared, "denseShared");
        this.tiled = Objects.requireNonNull(tiled, "tiled");
        this.slidingDecode = Objects.requireNonNull(slidingDecode, "slidingDecode");
    }

    AccelTensor denseSharedAttention(AccelTensor q, AccelTensor k, AccelTensor v,
            SharedKvState sharedKvState, ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            int layerIdx, int startPos, int numQHeads, int numKVHeads, int headDim,
            float scale, boolean causal, float softCap, MemorySegment attentionContextBuffer) {
        return denseShared.compute(q, k, v, sharedKvState, config, modelPolicy, layerIdx, startPos, numQHeads,
                numKVHeads, headDim, scale, causal, softCap, attentionContextBuffer);
    }

    AccelTensor tiledAttention(AccelTensor q, KVCacheManager.KVCacheSession kvSession, int kvLayerIdx,
            int startPos, int numHeads, int numKVHeads, int headDim, float scale, boolean causal, float softCap,
            ModelConfig config, FlashAttentionModelPolicy modelPolicy, int layerIdx,
            MemorySegment attentionContextBuffer) {
        return tiled.compute(q, kvSession, kvLayerIdx, startPos, numHeads, numKVHeads, headDim, scale, causal,
                softCap, config, modelPolicy, layerIdx, attentionContextBuffer);
    }

    AccelTensor slidingDecodeAttention(AccelTensor q, KVCacheManager.KVCacheSession kvSession,
            int layerIdx, int kvLayerIdx, int startPos, int numHeads, int numKVHeads, int headDim, float scale,
            float softCap, ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            MemorySegment attentionContextBuffer) {
        return slidingDecode.compute(q, kvSession, layerIdx, kvLayerIdx, startPos, numHeads, numKVHeads, headDim,
                scale, softCap, config, modelPolicy, attentionContextBuffer);
    }

}
