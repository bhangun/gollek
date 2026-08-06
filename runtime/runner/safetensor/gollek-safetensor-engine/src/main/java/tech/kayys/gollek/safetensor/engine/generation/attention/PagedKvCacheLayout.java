/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;

final class PagedKvCacheLayout {
    private final int numHeads;
    private final int headDim;
    private final int blockSize;
    private final long sourceHeadStride;
    private final long sourceTokenStride;
    private final long scaleStride;
    private final long blockElements;
    private final long blockHeadStride;

    private PagedKvCacheLayout(int numHeads, int headDim, int blockSize,
            long sourceHeadStride, long sourceTokenStride, long scaleStride) {
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.blockSize = blockSize;
        this.sourceHeadStride = sourceHeadStride;
        this.sourceTokenStride = sourceTokenStride;
        this.scaleStride = scaleStride;
        this.blockHeadStride = (long) blockSize * headDim;
        this.blockElements = (long) numHeads * blockHeadStride;
    }

    static PagedKvCacheLayout source(BlockManager blockManager, int numHeads, int headDim, int blockSize) {
        return new PagedKvCacheLayout(numHeads, headDim, blockSize,
                blockManager.getHeadStride(),
                blockManager.getTokenStride(),
                blockManager.getScaleStride());
    }

    static PagedKvCacheLayout packed(int numHeads, int headDim, int blockSize) {
        return new PagedKvCacheLayout(numHeads, headDim, blockSize, 0, 0, blockSize);
    }

    int tokenIndexInBlock(int absoluteToken) {
        return absoluteToken % blockSize;
    }

    long sourceElement(int headIdx, int tokenIdxInBlock) {
        return (long) headIdx * sourceHeadStride + (long) tokenIdxInBlock * sourceTokenStride;
    }

    long sourceByteOffset(int headIdx, int tokenIdxInBlock) {
        return sourceElement(headIdx, tokenIdxInBlock) * Float.BYTES;
    }

    long scaleIndex(int headIdx, int tokenIdxInBlock) {
        return (long) headIdx * scaleStride + tokenIdxInBlock;
    }

    long tokenMajorElement(int tokenIdx, int headIdx) {
        return ((long) tokenIdx * numHeads + headIdx) * headDim;
    }

    long blockMajorElement(int blockIdx, int headIdx, int tokenIdxInBlock) {
        return (long) blockIdx * blockElements
                + (long) headIdx * blockHeadStride
                + (long) tokenIdxInBlock * headDim;
    }

    long blockElements() {
        return blockElements;
    }
}
