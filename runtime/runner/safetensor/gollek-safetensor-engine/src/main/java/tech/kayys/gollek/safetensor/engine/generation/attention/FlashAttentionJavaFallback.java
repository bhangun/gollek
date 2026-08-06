/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;

/**
 * CPU fallback attention implementation used when a native direct attention
 * backend is unavailable or declines a shape.
 */
final class FlashAttentionJavaFallback {
    private FlashAttentionJavaFallback() {
    }

    static AccelTensor denseSharedAttention(AccelTensor q, AccelTensor k, AccelTensor v,
            ModelConfig config, int layerIdx, int startPos, int numQHeads, int numKVHeads, int headDim,
            float scale, boolean causal, float softCap) {
        return denseSharedAttention(q, k, v, config, layerIdx, startPos, numQHeads, numKVHeads, headDim,
                scale, causal, softCap, null);
    }

    static AccelTensor denseSharedAttention(AccelTensor q, AccelTensor k, AccelTensor v,
            ModelConfig config, int layerIdx, int startPos, int numQHeads, int numKVHeads, int headDim,
            float scale, boolean causal, float softCap, MemorySegment attentionContextBuffer) {
        // IMPORTANT: StridedKeyValueSource holds strong references to k and v (not just raw
        // MemorySegments) so that the backing Arena.ofAuto() cannot be GC-collected while the
        // long-running attention compute loop is still reading from those segments.
        FlashAttentionDenseFallbackLoop.KeyValueSource source = new StridedKeyValueSource(
                k,
                v,
                Math.toIntExact(k.size(1)),
                k.stride()[0],
                k.stride()[1],
                k.stride()[2],
                v.stride()[0],
                v.stride()[1],
                v.stride()[2]);
        try {
            return FlashAttentionDenseFallbackLoop.compute(
                    q, source, config, layerIdx, startPos, numQHeads, numKVHeads, headDim, scale, causal, softCap,
                    attentionContextBuffer);
        } finally {
            java.lang.ref.Reference.reachabilityFence(q);
            java.lang.ref.Reference.reachabilityFence(k);
            java.lang.ref.Reference.reachabilityFence(v);
        }
    }

    static AccelTensor denseCachedAttention(AccelTensor q, KVCacheManager.KVCacheSession kvSession, int kvLayerIdx,
            int startPos, int numHeads, int numKVHeads, int headDim, float scale, boolean causal, float softCap,
            ModelConfig config, int layerIdx) {
        return denseCachedAttention(q, kvSession, kvLayerIdx, startPos, numHeads, numKVHeads, headDim,
                scale, causal, softCap, config, layerIdx, null);
    }

    static AccelTensor denseCachedAttention(AccelTensor q, KVCacheManager.KVCacheSession kvSession, int kvLayerIdx,
            int startPos, int numHeads, int numKVHeads, int headDim, float scale, boolean causal, float softCap,
            ModelConfig config, int layerIdx, MemorySegment attentionContextBuffer) {
        BlockManager blockManager = kvSession.blockManager();
        long seqLen = q.size(1);
        int totalTokens = startPos + (int) seqLen;
        long gatherElements = (long) totalTokens * numKVHeads * headDim;

        FlashAttentionKvScratch.KvPools gatheredPools =
                FlashAttentionKvScratch.projectionScratchPools(kvSession, gatherElements,
                        "Dense cached attention gathered KV scratch");
        PagedKvCacheIO.gather(blockManager, kvSession, kvLayerIdx, totalTokens, numKVHeads, headDim,
                gatheredPools.key(), gatheredPools.value());
        return denseCachedAttention(q, gatheredPools.key(), gatheredPools.value(), startPos, numHeads, numKVHeads,
                headDim, scale, causal, softCap, config, layerIdx, attentionContextBuffer);
    }

    private static AccelTensor denseCachedAttention(AccelTensor q, MemorySegment kSeg, MemorySegment vSeg,
            int startPos, int numQHeads, int numKVHeads, int headDim, float scale, boolean causal, float softCap,
            ModelConfig config, int layerIdx, MemorySegment attentionContextBuffer) {
        FlashAttentionDenseFallbackLoop.KeyValueSource source = new GatheredKeyValueSource(
                kSeg, vSeg, startPos + Math.toIntExact(q.size(1)), numKVHeads, headDim);
        try {
            return FlashAttentionDenseFallbackLoop.compute(
                    q, source, config, layerIdx, startPos, numQHeads, numKVHeads, headDim, scale, causal, softCap,
                    attentionContextBuffer);
        } finally {
            java.lang.ref.Reference.reachabilityFence(q);
        }
    }

    private record StridedKeyValueSource(
            AccelTensor keyTensor,
            AccelTensor valueTensor,
            int totalTokens,
            long keyBatchStride,
            long keyTokenStride,
            long keyHeadStride,
            long valueBatchStride,
            long valueTokenStride,
            long valueHeadStride) implements FlashAttentionDenseFallbackLoop.KeyValueSource {

        @Override
        public MemorySegment keySegment() {
            // Accessing via the tensor (not a stored raw segment) keeps the tensor reachable
            return keyTensor.dataSegment();
        }

        @Override
        public MemorySegment valueSegment() {
            return valueTensor.dataSegment();
        }

        @Override
        public long keyOffset(int batch, int token, int kvHeadIdx) {
            return (long) batch * keyBatchStride + (long) token * keyTokenStride + (long) kvHeadIdx * keyHeadStride;
        }

        @Override
        public long valueOffset(int batch, int token, int kvHeadIdx) {
            return (long) batch * valueBatchStride + (long) token * valueTokenStride
                    + (long) kvHeadIdx * valueHeadStride;
        }
    }

    private record GatheredKeyValueSource(
            MemorySegment keySegment,
            MemorySegment valueSegment,
            int totalTokens,
            int numKVHeads,
            int headDim) implements FlashAttentionDenseFallbackLoop.KeyValueSource {

        @Override
        public long keyOffset(int batch, int token, int kvHeadIdx) {
            return ((long) token * numKVHeads + kvHeadIdx) * headDim;
        }

        @Override
        public long valueOffset(int batch, int token, int kvHeadIdx) {
            return ((long) token * numKVHeads + kvHeadIdx) * headDim;
        }
    }
}
