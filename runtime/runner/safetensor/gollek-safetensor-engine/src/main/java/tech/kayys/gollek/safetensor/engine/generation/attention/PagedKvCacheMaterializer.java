/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;

/**
 * Materializes paged KV cache layouts into temporary dense or packed buffers
 * expected by fallback and Metal attention kernels.
 */
final class PagedKvCacheMaterializer {
    private PagedKvCacheMaterializer() {
    }

    static void packDenseSharedKvIntoTemporaryPagedPool(AccelTensor key, AccelTensor value,
            int numKVHeads, int headDim, MemorySegment packedKey, MemorySegment packedValue) {
        long batch = key.size(0);
        int totalTokens = Math.toIntExact(key.size(1));
        MemorySegment keySeg = key.dataSegment();
        MemorySegment valueSeg = value.dataSegment();
        long keyStride0 = key.stride()[0];
        long keyStride1 = key.stride()[1];
        long keyStride2 = key.stride()[2];
        long valueStride0 = value.stride()[0];
        long valueStride1 = value.stride()[1];
        long valueStride2 = value.stride()[2];
        long perBatchBlockElements = (long) numKVHeads * totalTokens * headDim;

        for (int b = 0; b < batch; b++) {
            long batchBase = b * perBatchBlockElements;
            for (int tok = 0; tok < totalTokens; tok++) {
                for (int h = 0; h < numKVHeads; h++) {
                    long srcKElement = (long) b * keyStride0 + (long) tok * keyStride1 + (long) h * keyStride2;
                    long srcVElement = (long) b * valueStride0 + (long) tok * valueStride1 + (long) h * valueStride2;
                    long dstElement = batchBase + (long) h * totalTokens * headDim + (long) tok * headDim;
                    long bytes = (long) headDim * Float.BYTES;
                    MemorySegment.copy(keySeg, srcKElement * Float.BYTES, packedKey, dstElement * Float.BYTES, bytes);
                    MemorySegment.copy(valueSeg, srcVElement * Float.BYTES, packedValue, dstElement * Float.BYTES,
                            bytes);
                }
            }
        }
    }

    static MaterializedKvPools materializeBlocksForMetal(BlockManager blockManager,
            KVCacheManager.KVCacheSession kvSession, List<Integer> blocks,
            int numKVHeads, int headDim, Arena arena) {
        int blockSize = kvSession.tokensPerBlock();
        PagedKvCacheLayout layout = PagedKvCacheLayout.source(blockManager, numKVHeads, headDim, blockSize);
        long totalElements = (long) blocks.size() * layout.blockElements();
        MemorySegment packedK = arena.allocate(totalElements * Float.BYTES, 64);
        MemorySegment packedV = arena.allocate(totalElements * Float.BYTES, 64);
        BlockManager.KvStorageType storageType = blockManager.getStorageType();

        for (int localBlockIdx = 0; localBlockIdx < blocks.size(); localBlockIdx++) {
            int physBlockIdx = blocks.get(localBlockIdx);
            MemorySegment srcKBlock = blockManager.getKBlock(physBlockIdx);
            MemorySegment srcVBlock = blockManager.getVBlock(physBlockIdx);
            MemorySegment srcKScaleBlock = blockManager.getKScaleBlock(physBlockIdx);
            MemorySegment srcVScaleBlock = blockManager.getVScaleBlock(physBlockIdx);

            for (int h = 0; h < numKVHeads; h++) {
                for (int tok = 0; tok < blockSize; tok++) {
                    long srcElement = layout.sourceElement(h, tok);
                    long scaleIndex = layout.scaleIndex(h, tok);
                    long dstElement = layout.blockMajorElement(localBlockIdx, h, tok);
                    PagedKvCacheQuantization.writeVectorAsFloat(storageType, srcKBlock, srcKScaleBlock, srcElement,
                            scaleIndex, packedK, dstElement, headDim);
                    PagedKvCacheQuantization.writeVectorAsFloat(storageType, srcVBlock, srcVScaleBlock, srcElement,
                            scaleIndex, packedV, dstElement, headDim);
                }
            }
        }

        return new MaterializedKvPools(packedK, packedV);
    }

    static void packRangeIntoTemporaryPagedPool(BlockManager blockManager, KVCacheManager.KVCacheSession kvSession,
            int kvLayerIdx, int tokenStart, int tokenEnd, int numKVHeads, int headDim, int blockSize,
            MemorySegment packedK, MemorySegment packedV) {
        if (tokenEnd <= tokenStart) {
            return;
        }
        int sourceBlockSize = kvSession.tokensPerBlock();
        PagedKvCacheLayout sourceLayout = PagedKvCacheLayout.source(blockManager, numKVHeads, headDim,
                sourceBlockSize);
        PagedKvCacheLayout packedLayout = PagedKvCacheLayout.packed(numKVHeads, headDim, blockSize);
        BlockManager.KvStorageType storageType = blockManager.getStorageType();

        for (int runStart = tokenStart; runStart < tokenEnd;) {
            int srcBlockIdx = kvSession.getBlockForToken(kvLayerIdx, runStart);
            int runEnd = Math.min(tokenEnd, nextBlockStart(runStart, sourceBlockSize));
            MemorySegment srcKBlock = blockManager.getKBlock(srcBlockIdx);
            MemorySegment srcVBlock = blockManager.getVBlock(srcBlockIdx);
            MemorySegment srcKScaleBlock = blockManager.getKScaleBlock(srcBlockIdx);
            MemorySegment srcVScaleBlock = blockManager.getVScaleBlock(srcBlockIdx);

            for (int tok = runStart; tok < runEnd; tok++) {
                int srcTokInBlock = sourceLayout.tokenIndexInBlock(tok);
                int localTok = tok - tokenStart;
                int dstBlockIdx = localTok / blockSize;
                int dstTokInBlock = localTok % blockSize;

                for (int h = 0; h < numKVHeads; h++) {
                    long srcElement = sourceLayout.sourceElement(h, srcTokInBlock);
                    long scaleIndex = sourceLayout.scaleIndex(h, srcTokInBlock);
                    long dstElement = packedLayout.blockMajorElement(dstBlockIdx, h, dstTokInBlock);
                    if (storageType == BlockManager.KvStorageType.FP32) {
                        long bytes = (long) headDim * Float.BYTES;
                        MemorySegment.copy(srcKBlock, srcElement * Float.BYTES, packedK,
                                dstElement * Float.BYTES, bytes);
                        MemorySegment.copy(srcVBlock, srcElement * Float.BYTES, packedV,
                                dstElement * Float.BYTES, bytes);
                    } else {
                        PagedKvCacheQuantization.writeVectorAsFloat(storageType, srcKBlock, srcKScaleBlock,
                                srcElement, scaleIndex, packedK, dstElement, headDim);
                        PagedKvCacheQuantization.writeVectorAsFloat(storageType, srcVBlock, srcVScaleBlock,
                                srcElement, scaleIndex, packedV, dstElement, headDim);
                    }
                }
            }
            runStart = runEnd;
        }
    }

    private static int nextBlockStart(int token, int blockSize) {
        return ((token / blockSize) + 1) * blockSize;
    }

    record MaterializedKvPools(MemorySegment kPool, MemorySegment vPool) {
    }
}
