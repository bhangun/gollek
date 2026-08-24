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

record FlashAttentionPlan(ModelConfig config,
        FlashAttentionModelPolicy modelPolicy,
        int layerIdx,
        int startPos,
        int seqLen,
        FlashAttentionHeadLayout headLayout,
        FlashAttentionNormalizationPolicy normalizationPolicy,
        boolean alternativeAttention,
        float attentionScale,
        float attentionSoftCap,
        int kvLayerIdx,
        FlashAttentionKvCacheStage.State kvState) {

    static FlashAttentionPlan resolve(AttentionInput in, FlashAttentionKvCacheStage kvCache) {
        return resolve(in, kvCache, FlashAttentionNormalizationOptions.fromSystemProperties());
    }

    static FlashAttentionPlan resolve(AttentionInput in, FlashAttentionKvCacheStage kvCache,
            FlashAttentionNormalizationOptions normalizationOptions) {
        ModelConfig config = in.config;
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(in.arch, config);
        int layerIdx = in.layerIdx;
        int startPos = in.startPos;
        int seqLen = (int) in.x.size(1);
        FlashAttentionHeadLayout headLayout = FlashAttentionHeadLayout.resolve(in, config, layerIdx, modelPolicy);
        FlashAttentionNormalizationPolicy normalizationPolicy =
                FlashAttentionNormalizationPolicy.resolve(in.arch, config, modelPolicy, normalizationOptions);
        int kvLayerIdx = config.sharedKvSourceLayer(layerIdx);
        return new FlashAttentionPlan(
                config,
                modelPolicy,
                layerIdx,
                startPos,
                seqLen,
                headLayout,
                normalizationPolicy,
                config.usesAlternativeAttentionForLayer(layerIdx),
                normalizationPolicy.attentionScale(),
                modelPolicy.resolveAttentionSoftCap(),
                kvLayerIdx,
                kvCache.resolveState(in, config, layerIdx, kvLayerIdx));
    }

    int numQueryHeads() {
        return headLayout.numQueryHeads();
    }

    int numKeyValueHeads() {
        return headLayout.numKeyValueHeads();
    }

    int headDim() {
        return headLayout.headDim();
    }

    boolean sharedKv() {
        return kvState.sharedKv();
    }

    boolean useDenseSharedKvState() {
        return kvState.useDenseSharedKvState();
    }

    SharedKvState sharedKvState() {
        return kvState.sharedKvState();
    }

    boolean addOneRmsNorm() {
        return normalizationPolicy.addOneToRmsNormWeight();
    }

    AccelTensor applyOutputStage(FlashAttentionOutputStage outputStage, AttentionInput in,
            AccelTensor attentionOutput) {
        return outputStage.project(in, attentionOutput, seqLen, headLayout, config, modelPolicy, addOneRmsNorm());
    }

    FlashAttentionDispatchRequest dispatchRequest(AccelTensor query, AccelTensor key, AccelTensor value,
            KVCacheManager.KVCacheSession kvSession, boolean causal, MemorySegment attentionContextBuffer) {
        return new FlashAttentionDispatchRequest(
                query, key, value, kvSession, sharedKvState(), config, modelPolicy, layerIdx, kvLayerIdx, startPos,
                seqLen, numQueryHeads(), numKeyValueHeads(), headDim(), attentionScale, causal, attentionSoftCap,
                attentionContextBuffer, useDenseSharedKvState());
    }
}
