/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

final class PagedKvCacheQuantization {
    private PagedKvCacheQuantization() {
    }

    static void writeVector(BlockManager.KvStorageType storageType,
            MemorySegment srcSeg, long srcFloatIndex,
            MemorySegment dstSeg, long dstElementIndex,
            MemorySegment scaleSeg, long scaleIndex,
            int headDim) {
        switch (storageType) {
            case FP32 -> MemorySegment.copy(srcSeg, ValueLayout.JAVA_FLOAT, srcFloatIndex * Float.BYTES,
                    dstSeg, ValueLayout.JAVA_FLOAT, dstElementIndex * Float.BYTES, headDim);
            case INT8 -> quantizeVectorToInt8(srcSeg, srcFloatIndex, dstSeg, dstElementIndex,
                    scaleSeg, scaleIndex, headDim);
            case INT4 -> quantizeVectorToInt4(srcSeg, srcFloatIndex, dstSeg, dstElementIndex,
                    scaleSeg, scaleIndex, headDim);
        }
    }

    static void writeVectorAsFloat(BlockManager.KvStorageType storageType,
            MemorySegment srcBlock, MemorySegment scaleBlock,
            long srcElement, long scaleIndex,
            MemorySegment dst, long dstElement,
            int headDim) {
        switch (storageType) {
            case FP32 -> {
                long bytes = (long) headDim * Float.BYTES;
                MemorySegment.copy(srcBlock, srcElement * Float.BYTES, dst, dstElement * Float.BYTES, bytes);
            }
            case INT8 -> {
                float scale = scale(scaleBlock, scaleIndex);
                for (int d = 0; d < headDim; d++) {
                    float value = srcBlock.getAtIndex(ValueLayout.JAVA_BYTE, srcElement + d) * scale;
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstElement + d, value);
                }
            }
            case INT4 -> {
                float scale = scale(scaleBlock, scaleIndex);
                for (int d = 0; d < headDim; d++) {
                    float value = readPackedSignedInt4(srcBlock, srcElement + d) * scale;
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstElement + d, value);
                }
            }
        }
    }

    static int readPackedSignedInt4(MemorySegment srcSeg, long srcElementIndex) {
        long byteIndex = srcElementIndex >>> 1;
        int packed = Byte.toUnsignedInt(srcSeg.getAtIndex(ValueLayout.JAVA_BYTE, byteIndex));
        int nibble = (srcElementIndex & 1L) == 0L ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);
        return nibble - 8;
    }

    private static void quantizeVectorToInt8(MemorySegment srcSeg, long srcFloatIndex,
            MemorySegment dstSeg, long dstElementIndex,
            MemorySegment scaleSeg, long scaleIndex,
            int headDim) {
        float scale = vectorScale(srcSeg, srcFloatIndex, headDim, 127.0f);
        writeScale(scaleSeg, scaleIndex, scale);

        for (int d = 0; d < headDim; d++) {
            float value = srcSeg.getAtIndex(ValueLayout.JAVA_FLOAT, srcFloatIndex + d);
            int quantized = Math.round(value / scale);
            quantized = Math.max(-127, Math.min(127, quantized));
            dstSeg.setAtIndex(ValueLayout.JAVA_BYTE, dstElementIndex + d, (byte) quantized);
        }
    }

    private static void quantizeVectorToInt4(MemorySegment srcSeg, long srcFloatIndex,
            MemorySegment dstSeg, long dstElementIndex,
            MemorySegment scaleSeg, long scaleIndex,
            int headDim) {
        float scale = vectorScale(srcSeg, srcFloatIndex, headDim, 7.0f);
        writeScale(scaleSeg, scaleIndex, scale);

        for (int d = 0; d < headDim; d++) {
            float value = srcSeg.getAtIndex(ValueLayout.JAVA_FLOAT, srcFloatIndex + d);
            int quantized = Math.round(value / scale);
            quantized = Math.max(-8, Math.min(7, quantized));
            writePackedSignedInt4(dstSeg, dstElementIndex + d, quantized);
        }
    }

    private static float vectorScale(MemorySegment srcSeg, long srcFloatIndex, int headDim, float maxQuantized) {
        float absmax = 0.0f;
        for (int d = 0; d < headDim; d++) {
            float value = srcSeg.getAtIndex(ValueLayout.JAVA_FLOAT, srcFloatIndex + d);
            float abs = Math.abs(value);
            if (abs > absmax) {
                absmax = abs;
            }
        }
        return absmax == 0.0f ? 1.0f : absmax / maxQuantized;
    }

    private static void writeScale(MemorySegment scaleSeg, long scaleIndex, float scale) {
        if (scaleSeg != null) {
            scaleSeg.setAtIndex(ValueLayout.JAVA_FLOAT, scaleIndex, scale);
        }
    }

    private static float scale(MemorySegment scaleSeg, long scaleIndex) {
        return scaleSeg == null ? 1.0f : scaleSeg.getAtIndex(ValueLayout.JAVA_FLOAT, scaleIndex);
    }

    private static void writePackedSignedInt4(MemorySegment dstSeg, long dstElementIndex, int quantized) {
        long byteIndex = dstElementIndex >>> 1;
        int stored = quantized + 8;
        int existing = Byte.toUnsignedInt(dstSeg.getAtIndex(ValueLayout.JAVA_BYTE, byteIndex));
        int packed = (dstElementIndex & 1L) == 0L
                ? ((existing & 0xF0) | (stored & 0x0F))
                : ((existing & 0x0F) | ((stored & 0x0F) << 4));
        dstSeg.setAtIndex(ValueLayout.JAVA_BYTE, byteIndex, (byte) packed);
    }
}
