/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.function.Supplier;

final class FlashAttentionMetalTiledAttention {
    private final Supplier<FlashAttentionRoutingPolicy> routingPolicy;
    private final FlashAttentionMetalPagedAttention pagedAttention;
    private final FlashAttentionMetalFa4GatheredAttention fa4Gathered;
    private final PagedAttentionVectorOptions pagedAttentionOptions;

    FlashAttentionMetalTiledAttention(Supplier<FlashAttentionRoutingPolicy> routingPolicy,
            FlashAttentionMetalPagedAttention pagedAttention,
            FlashAttentionMetalFa4GatheredAttention fa4Gathered) {
        this(routingPolicy, pagedAttention, fa4Gathered, PagedAttentionVectorOptions.defaults());
    }

    FlashAttentionMetalTiledAttention(Supplier<FlashAttentionRoutingPolicy> routingPolicy,
            FlashAttentionMetalPagedAttention pagedAttention,
            FlashAttentionMetalFa4GatheredAttention fa4Gathered,
            PagedAttentionVectorOptions pagedAttentionOptions) {
        this.routingPolicy = routingPolicy;
        this.pagedAttention = pagedAttention;
        this.fa4Gathered = fa4Gathered;
        this.pagedAttentionOptions =
                pagedAttentionOptions == null ? PagedAttentionVectorOptions.defaults() : pagedAttentionOptions;
    }

    AccelTensor compute(AccelTensor q, KVCacheManager.KVCacheSession kvSession, int kvLayerIdx,
            int startPos, int numHeads, int numKVHeads, int headDim, float scale, boolean causal, float softCap,
            ModelConfig config, FlashAttentionModelPolicy modelPolicy, int layerIdx,
            MemorySegment attentionContextBuffer) {
        BlockManager blockManager = kvSession.blockManager();
        long batch = q.size(0);
        long seqLen = q.size(1);
        int totalTokens = startPos + (int) seqLen;
        List<Integer> blocks = kvSession.getBlockIndices(kvLayerIdx);
        FlashAttentionRoutingPolicy routing = routingPolicy();
        boolean canUseFa4Path = routing.canUseFa4Attention(softCap);
        boolean slidingLayer = config != null && config.isSlidingAttentionLayer(layerIdx) && config.hasSlidingWindow();
        boolean preferPagedMetalFirst = routing.preferPagedMetalAttentionBeforeFa4(modelPolicy, (int) seqLen,
                totalTokens);

        if (preferPagedMetalFirst && !kvSession.isQuantized()) {
            AccelTensor pagedOut = pagedAttention.tryCompute(q, kvSession, blockManager, blocks, kvLayerIdx,
                    startPos, numHeads, numKVHeads, headDim, scale, causal, softCap, config, layerIdx, totalTokens,
                    batch, seqLen, slidingLayer, null, modelPolicy, attentionContextBuffer);
            if (pagedOut != null) {
                DirectInferenceProfiler.recordAttentionPath(
                        routing.pagedAttentionPathName(slidingLayer, seqLen, totalTokens, true));
                return pagedOut;
            }
        }

        try (Arena arena = Arena.ofConfined()) {
            if (!preferPagedMetalFirst && canUseFa4Path) {
                AccelTensor fa4Out = fa4Gathered.tryCompute(q, kvSession, blockManager, kvLayerIdx, totalTokens,
                        numHeads, numKVHeads, headDim, scale, causal, softCap, batch, seqLen, arena,
                        "fa4_gathered", attentionContextBuffer);
                if (fa4Out != null) {
                    return fa4Out;
                }
            }
            AccelTensor pagedOut = pagedAttention.tryCompute(q, kvSession, blockManager, blocks, kvLayerIdx,
                    startPos, numHeads, numKVHeads, headDim, scale, causal, softCap, config, layerIdx, totalTokens,
                    batch, seqLen, slidingLayer, arena, modelPolicy, attentionContextBuffer);
            if (pagedOut != null) {
                DirectInferenceProfiler.recordAttentionPath(
                        routing.pagedAttentionPathName(slidingLayer, seqLen, totalTokens, false));
                return pagedOut;
            }
            if (canUseFa4Path) {
                AccelTensor fa4Out = fa4Gathered.tryCompute(q, kvSession, blockManager, kvLayerIdx, totalTokens,
                        numHeads, numKVHeads, headDim, scale, causal, softCap, batch, seqLen, arena,
                        "fa4_gathered_after_paged", attentionContextBuffer);
                if (fa4Out != null) {
                    return fa4Out;
                }
            }
            DirectInferenceProfiler.recordAttentionPath("paged_java");
            return PagedAttentionVectorAPI.compute(q, null, kvSession, kvLayerIdx, kvLayerIdx, startPos, numHeads,
                    numKVHeads, headDim, scale, causal, softCap, pagedAttentionOptions, attentionContextBuffer);
        }
    }

    private FlashAttentionRoutingPolicy routingPolicy() {
        return routingPolicy.get();
    }
}
