/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

final class PagedAttentionVectorMath {
    private PagedAttentionVectorMath() {
    }

    static float score(BlockManager.KvStorageType storageType, MemorySegment qSeg, long qOffset,
            MemorySegment keyBlock, long keyElementOffset, long keyByteOffset, MemorySegment keyScaleBlock,
            long scaleIndex, int headDim, float scale) {
        return switch (storageType) {
            case INT8 -> dotProductInt8(qSeg, qOffset, keyBlock, keyElementOffset, keyScaleBlock, scaleIndex, headDim)
                    * scale;
            case INT4 -> dotProductInt4(qSeg, qOffset, keyBlock, keyElementOffset, keyScaleBlock, scaleIndex, headDim)
                    * scale;
            case FP32 -> AttentionFp32VectorMath.dotProduct(qSeg, qOffset, keyBlock, keyByteOffset, headDim) * scale;
        };
    }

    static void updateAccumulator(BlockManager.KvStorageType storageType, float[] acc, MemorySegment valueBlock,
            long valueElementOffset, long valueByteOffset, MemorySegment valueScaleBlock, long scaleIndex,
            float previousWeight, float currentWeight, int headDim) {
        switch (storageType) {
            case INT8 -> updateAccumulatorInt8(acc, valueBlock, valueElementOffset, valueScaleBlock, scaleIndex,
                    previousWeight, currentWeight, headDim);
            case INT4 -> updateAccumulatorInt4(acc, valueBlock, valueElementOffset, valueScaleBlock, scaleIndex,
                    previousWeight, currentWeight, headDim);
            case FP32 -> AttentionFp32VectorMath.updateAccumulator(acc, valueBlock, valueByteOffset, previousWeight,
                    currentWeight, headDim);
        }
    }

    static void writeNormalizedAccumulator(MemorySegment out, long outIndex, float[] acc, float invL, int headDim) {
        AttentionFp32VectorMath.writeNormalizedAccumulator(out, outIndex, acc, invL, headDim);
    }

    private static float dotProductInt8(MemorySegment q, long qOffset, MemorySegment k, long kElementOffset,
            MemorySegment scaleSeg, long scaleIndex, int dim) {
        float scale = scaleSeg == null ? 1.0f : scaleSeg.getAtIndex(ValueLayout.JAVA_FLOAT, scaleIndex);
        float result = 0.0f;
        for (int j = 0; j < dim; j++) {
            result += q.getAtIndex(ValueLayout.JAVA_FLOAT, (qOffset / Float.BYTES) + j)
                    * (k.getAtIndex(ValueLayout.JAVA_BYTE, kElementOffset + j) * scale);
        }
        return result;
    }

    private static void updateAccumulatorInt8(float[] acc, MemorySegment vSeg, long vElementOffset,
            MemorySegment scaleSeg, long scaleIndex, float previousWeight, float currentWeight, int dim) {
        float scale = scaleSeg == null ? 1.0f : scaleSeg.getAtIndex(ValueLayout.JAVA_FLOAT, scaleIndex);
        for (int j = 0; j < dim; j++) {
            acc[j] = acc[j] * previousWeight
                    + (vSeg.getAtIndex(ValueLayout.JAVA_BYTE, vElementOffset + j) * scale) * currentWeight;
        }
    }

    private static float dotProductInt4(MemorySegment q, long qOffset, MemorySegment k, long kElementOffset,
            MemorySegment scaleSeg, long scaleIndex, int dim) {
        float scale = scaleSeg == null ? 1.0f : scaleSeg.getAtIndex(ValueLayout.JAVA_FLOAT, scaleIndex);
        float result = 0.0f;
        for (int j = 0; j < dim; j++) {
            result += q.getAtIndex(ValueLayout.JAVA_FLOAT, (qOffset / Float.BYTES) + j)
                    * (PagedKvCacheQuantization.readPackedSignedInt4(k, kElementOffset + j) * scale);
        }
        return result;
    }

    private static void updateAccumulatorInt4(float[] acc, MemorySegment vSeg, long vElementOffset,
            MemorySegment scaleSeg, long scaleIndex, float previousWeight, float currentWeight, int dim) {
        float scale = scaleSeg == null ? 1.0f : scaleSeg.getAtIndex(ValueLayout.JAVA_FLOAT, scaleIndex);
        for (int j = 0; j < dim; j++) {
            acc[j] = acc[j] * previousWeight
                    + (PagedKvCacheQuantization.readPackedSignedInt4(vSeg, vElementOffset + j) * scale) * currentWeight;
        }
    }
}
