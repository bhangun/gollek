/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.function.Supplier;

final class FlashAttentionMetalSlidingDecodeAttention {
    private final Supplier<MetalBinding> metalBinding;
    private final Supplier<FlashAttentionRoutingPolicy> routingPolicy;
    private final FlashAttentionMetalPagedAttention pagedAttention;
    private final PagedAttentionVectorOptions pagedAttentionOptions;

    FlashAttentionMetalSlidingDecodeAttention(Supplier<MetalBinding> metalBinding,
            Supplier<FlashAttentionRoutingPolicy> routingPolicy, FlashAttentionMetalPagedAttention pagedAttention) {
        this(metalBinding, routingPolicy, pagedAttention, PagedAttentionVectorOptions.defaults());
    }

    FlashAttentionMetalSlidingDecodeAttention(Supplier<MetalBinding> metalBinding,
            Supplier<FlashAttentionRoutingPolicy> routingPolicy, FlashAttentionMetalPagedAttention pagedAttention,
            PagedAttentionVectorOptions pagedAttentionOptions) {
        this.metalBinding = metalBinding;
        this.routingPolicy = routingPolicy;
        this.pagedAttention = pagedAttention;
        this.pagedAttentionOptions = pagedAttentionOptions == null ? PagedAttentionVectorOptions.defaults()
                : pagedAttentionOptions;
    }

    AccelTensor compute(AccelTensor q, KVCacheManager.KVCacheSession kvSession,
            int layerIdx, int kvLayerIdx, int startPos, int numHeads, int numKVHeads, int headDim, float scale,
            float softCap, ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            MemorySegment attentionContextBuffer) {
        FlashAttentionRoutingPolicy routing = routingPolicy();
        if (!routing.allowSlidingDecodeMetalAttentionBridge(config, modelPolicy, layerIdx, (int) q.size(1))) {
            return javaFallback(q, config, kvSession, layerIdx, kvLayerIdx, startPos, numHeads, numKVHeads, headDim,
                    scale, softCap, attentionContextBuffer);
        }
        MetalBinding binding = metalBinding();
        if (binding == null || !binding.isRuntimeActive()) {
            return javaFallback(q, config, kvSession, layerIdx, kvLayerIdx, startPos, numHeads, numKVHeads, headDim,
                    scale, softCap, attentionContextBuffer);
        }
        long batch = q.size(0);
        int seqLen = (int) q.size(1);
        int totalTokens = startPos + seqLen;
        int slidingWindow = config.getSlidingWindowSize();
        int contextStart = Math.max(0, totalTokens - slidingWindow);
        int contextTokens = totalTokens - contextStart;
        if (routing.shortDecodeUsesNativeAttention(contextTokens)) {
            DirectInferenceProfiler.recordAttentionPath("sliding_decode_native_short");
            return PagedAttentionVectorAPI.compute(q, config, kvSession, layerIdx, kvLayerIdx, startPos, numHeads,
                    numKVHeads, headDim, scale, true, softCap, pagedAttentionOptions, attentionContextBuffer);
        }
        int blockSize = kvSession.tokensPerBlock();
        int maxBlocks = Math.max(1, (contextTokens + blockSize - 1) / blockSize);
        long poolElements = (long) maxBlocks * numKVHeads * blockSize * headDim;
        int batchInt = Math.toIntExact(batch);

        if (routing.enableRawPagedSlidingDecodeAttention()) {
            AccelTensor rawPagedOut = pagedAttention.tryCompute(q, kvSession, kvSession.blockManager(),
                    kvSession.getBlockIndices(kvLayerIdx), kvLayerIdx, startPos,
                    numHeads, numKVHeads, headDim, scale, true, softCap, config, layerIdx,
                    totalTokens, batch, seqLen, true, null, modelPolicy, attentionContextBuffer);
            if (rawPagedOut != null) {
                DirectInferenceProfiler.recordAttentionPath("sliding_decode_paged_metal");
                return rawPagedOut;
            }
        }

        FlashAttentionKvScratch.KvPools packedPools = FlashAttentionKvScratch.projectionScratchPools(kvSession,
                poolElements,
                "Sliding decode packed KV scratch");
        MemorySegment packedK = packedPools.key();
        MemorySegment packedV = packedPools.value();
        PagedKvCacheMaterializer.packRangeIntoTemporaryPagedPool(kvSession.blockManager(), kvSession, kvLayerIdx,
                contextStart, totalTokens, numKVHeads, headDim, blockSize, packedK, packedV);

        ForwardWorkspace ws = kvSession.getWorkspace();
        ws.ensureAttentionMetadataCapacity(batchInt * maxBlocks, batchInt);
        MemorySegment blockTable = ws.getAttentionBlockTableSeg();
        MemorySegment contextLens = ws.getAttentionContextLensSeg();
        for (int b = 0; b < batchInt; b++) {
            for (int blk = 0; blk < maxBlocks; blk++) {
                blockTable.setAtIndex(ValueLayout.JAVA_INT, (long) b * maxBlocks + blk, blk);
            }
        }

        for (int b = 0; b < batchInt; b++) {
            contextLens.setAtIndex(ValueLayout.JAVA_INT, b, contextTokens);
        }

        AccelTensor out = FlashAttentionContextOutputBuffer.viewOrAllocate(attentionContextBuffer, q);
        boolean success = false;
        try {
            int result = numKVHeads == numHeads
                    ? binding.attention(
                            out.dataPtr(), q.dataPtr(), packedK, packedV,
                            blockTable, contextLens,
                            (int) batch, seqLen, numHeads, headDim,
                            blockSize, maxBlocks,
                            scale, 0, softCap)
                    : binding.attentionGqa(
                            out.dataPtr(), q.dataPtr(), packedK, packedV,
                            blockTable, contextLens,
                            (int) batch, seqLen, numHeads, numKVHeads, headDim,
                            blockSize, maxBlocks,
                            scale, 0, softCap);
            if (result == 0) {
                success = true;
                DirectInferenceProfiler.recordAttentionPath("sliding_decode_metal");
                return out;
            }
            return javaFallback(q, config, kvSession, layerIdx, kvLayerIdx, startPos, numHeads, numKVHeads,
                    headDim, scale, softCap, attentionContextBuffer);
        } finally {
            if (!success && !out.isClosed()) {
                out.close();
            }
        }
    }

    private AccelTensor javaFallback(AccelTensor q, ModelConfig config, KVCacheManager.KVCacheSession kvSession,
            int layerIdx, int kvLayerIdx, int startPos, int numHeads, int numKVHeads, int headDim, float scale,
            float softCap, MemorySegment attentionContextBuffer) {
        DirectInferenceProfiler.recordAttentionPath("sliding_decode_java");
        return PagedAttentionVectorAPI.compute(q, config, kvSession, layerIdx, kvLayerIdx, startPos, numHeads,
                numKVHeads, headDim, scale, true, softCap, pagedAttentionOptions, attentionContextBuffer);
    }

    private MetalBinding metalBinding() {
        return metalBinding.get();
    }

    private FlashAttentionRoutingPolicy routingPolicy() {
        return routingPolicy.get();
    }
}
