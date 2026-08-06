/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;

import java.lang.foreign.MemorySegment;
import java.util.List;

/**
 * Copies projected attention keys and values between dense tensors and the
 * paged KV cache used by direct inference.
 */
final class PagedKvCacheIO {
    private PagedKvCacheIO() {
    }

    static void updateCache(AccelTensor key, AccelTensor value, KVCacheManager.KVCacheSession kvSession, int layerIdx,
            int startPos, int seqLen, int numHeads, int headDim) {
        BlockManager blockManager = kvSession.blockManager();
        MemorySegment keySeg = key.dataPtr();
        MemorySegment valueSeg = value.dataPtr();

        // Writes happen before kvCache.advance(...), so we must provision the
        // backing block tables for the whole write span up front.
        kvSession.ensureCapacity(startPos + seqLen);

        int tokensPerBlock = kvSession.tokensPerBlock();
        PagedKvCacheLayout layout = PagedKvCacheLayout.source(blockManager, numHeads, headDim, tokensPerBlock);

        BlockManager.KvStorageType storageType = blockManager.getStorageType();
        for (int s = 0; s < seqLen; s++) {
            int absolutePos = startPos + s;
            int blockIdx = kvSession.getBlockForToken(layerIdx, absolutePos);
            if (blockIdx < 0) {
                throw new IllegalStateException(
                        "Missing KV block for layer " + layerIdx + " token " + absolutePos
                                + " (startPos=" + startPos + ", seqLen=" + seqLen + ")");
            }
            int tokenIdxInBlock = layout.tokenIndexInBlock(absolutePos);

            MemorySegment keyBlock = blockManager.getKBlock(blockIdx);
            MemorySegment valueBlock = blockManager.getVBlock(blockIdx);
            MemorySegment keyScaleBlock = blockManager.getKScaleBlock(blockIdx);
            MemorySegment valueScaleBlock = blockManager.getVScaleBlock(blockIdx);

            for (int h = 0; h < numHeads; h++) {
                long srcOff = ((long) s * numHeads + h) * headDim;
                long dstElement = layout.sourceElement(h, tokenIdxInBlock);
                long scaleIndex = layout.scaleIndex(h, tokenIdxInBlock);
                PagedKvCacheQuantization.writeVector(storageType, keySeg, srcOff, keyBlock, dstElement,
                        keyScaleBlock, scaleIndex, headDim);
                PagedKvCacheQuantization.writeVector(storageType, valueSeg, srcOff, valueBlock, dstElement,
                        valueScaleBlock, scaleIndex, headDim);
            }
        }
    }

    static void gather(BlockManager blockManager, KVCacheManager.KVCacheSession kvSession, int kvLayerIdx,
            int totalTokens, int numKVHeads, int headDim, MemorySegment keyOut, MemorySegment valueOut) {
        List<Integer> blocks = kvSession.getBlockIndices(kvLayerIdx);
        int blockSize = kvSession.tokensPerBlock();
        PagedKvCacheLayout layout = PagedKvCacheLayout.source(blockManager, numKVHeads, headDim, blockSize);
        BlockManager.KvStorageType storageType = blockManager.getStorageType();
        int usedBlocks = usedBlockCount(totalTokens, blockSize, blocks.size());
        for (int blk = 0; blk < usedBlocks; blk++) {
            int phys = blocks.get(blk);
            int tokenStart = blk * blockSize;
            int tokenEnd = Math.min(totalTokens, tokenStart + blockSize);
            MemorySegment keyBlock = blockManager.getKBlock(phys);
            MemorySegment valueBlock = blockManager.getVBlock(phys);
            MemorySegment keyScaleBlock = blockManager.getKScaleBlock(phys);
            MemorySegment valueScaleBlock = blockManager.getVScaleBlock(phys);

            for (int tok = tokenStart; tok < tokenEnd; tok++) {
                int tokInBlock = layout.tokenIndexInBlock(tok);
                for (int h = 0; h < numKVHeads; h++) {
                    long srcElement = layout.sourceElement(h, tokInBlock);
                    long dstElement = layout.tokenMajorElement(tok, h);
                    long scaleIndex = layout.scaleIndex(h, tokInBlock);
                    PagedKvCacheQuantization.writeVectorAsFloat(storageType, keyBlock, keyScaleBlock, srcElement,
                            scaleIndex, keyOut, dstElement, headDim);
                    PagedKvCacheQuantization.writeVectorAsFloat(storageType, valueBlock, valueScaleBlock, srcElement,
                            scaleIndex, valueOut, dstElement, headDim);
                }
            }
        }
    }

    private static int usedBlockCount(int totalTokens, int blockSize, int availableBlocks) {
        if (totalTokens <= 0) {
            return 0;
        }
        int requiredBlocks = (totalTokens + blockSize - 1) / blockSize;
        return Math.min(requiredBlocks, availableBlocks);
    }
}
