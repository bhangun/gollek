/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;

import java.lang.foreign.MemorySegment;
import java.util.List;

record PagedAttentionVectorContext(
        AccelTensor queryTensor,
        MemorySegment querySegment,
        AccelTensor outputTensor,
        MemorySegment outputSegment,
        List<Integer> blockTable,
        BlockManager blockManager,
        PagedKvCacheLayout layout,
        BlockManager.KvStorageType storageType,
        int tokensPerBlock,
        int totalTokens,
        int numKvHeads,
        int headDim,
        float scale,
        boolean causal,
        float softCap,
        int slidingWindow,
        boolean debugProbeEnabled,
        int layerIdx) {

    static PagedAttentionVectorContext create(AccelTensor queryTensor, AccelTensor outputTensor,
            List<Integer> blockTable, BlockManager blockManager, PagedKvCacheLayout layout,
            BlockManager.KvStorageType storageType, int tokensPerBlock, int totalTokens, int numKvHeads, int headDim,
            float scale, boolean causal, float softCap, int slidingWindow, boolean debugProbeEnabled, int layerIdx) {
        return new PagedAttentionVectorContext(
                queryTensor,
                queryTensor.dataSegment(),
                outputTensor,
                outputTensor.dataSegment(),
                blockTable,
                blockManager,
                layout,
                storageType,
                tokensPerBlock,
                totalTokens,
                numKvHeads,
                headDim,
                scale,
                causal,
                softCap,
                slidingWindow,
                debugProbeEnabled,
                layerIdx);
    }

    PagedAttentionVectorQuery queryAt(int batchIndex, int qHeadIndex, int queryIndex) {
        int seqLenQ = Math.toIntExact(queryTensor.size(1));
        int queryStartPos = totalTokens - seqLenQ;
        int absolutePosition = queryStartPos + queryIndex;
        int minPosition = slidingWindow == Integer.MAX_VALUE
                ? 0
                : Math.max(0, absolutePosition - slidingWindow + 1);
        int kvHeadIndex = kvHeadIndex(qHeadIndex);
        boolean debugProbe = debugProbeEnabled
                && layerIdx == 0
                && batchIndex == 0
                && qHeadIndex == 0
                && queryIndex == seqLenQ - 1;
        return new PagedAttentionVectorQuery(
                batchIndex,
                qHeadIndex,
                queryIndex,
                kvHeadIndex,
                absolutePosition,
                minPosition,
                queryByteOffset(batchIndex, qHeadIndex, queryIndex),
                outputElementIndex(batchIndex, qHeadIndex, queryIndex),
                debugProbe);
    }

    private int kvHeadIndex(int qHeadIndex) {
        int qHeads = Math.toIntExact(queryTensor.size(2));
        int gqaGroup = qHeads / numKvHeads;
        return qHeadIndex / gqaGroup;
    }

    private long queryByteOffset(int batchIndex, int qHeadIndex, int queryIndex) {
        return tensorElementOffset(queryTensor, batchIndex, qHeadIndex, queryIndex) * Float.BYTES;
    }

    private long outputElementIndex(int batchIndex, int qHeadIndex, int queryIndex) {
        return tensorElementOffset(outputTensor, batchIndex, qHeadIndex, queryIndex);
    }

    private static long tensorElementOffset(AccelTensor tensor, int batchIndex, int qHeadIndex, int queryIndex) {
        long[] stride = tensor.stride();
        return (long) batchIndex * stride[0]
                + (long) queryIndex * stride[1]
                + (long) qHeadIndex * stride[2];
    }
}
