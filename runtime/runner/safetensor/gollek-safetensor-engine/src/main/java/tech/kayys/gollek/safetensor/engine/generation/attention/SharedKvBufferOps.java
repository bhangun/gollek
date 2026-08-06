/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import java.lang.foreign.MemorySegment;

final class SharedKvBufferOps {
    private SharedKvBufferOps() {
    }

    static AccelTensor viewTokenMajor(AccelTensor buffer, int tokenCount) {
        if (tokenCount == Math.toIntExact(buffer.size(1))) {
            return buffer;
        }
        return buffer.slice(1, 0, tokenCount);
    }

    static AccelTensor growTokenMajor(AccelTensor existing, int newCapacity, int numHeads, int headDim,
            int currentLength) {
        long batch = existing.size(0);
        AccelTensor grown = AccelTensor.zeros(batch, newCapacity, numHeads, headDim);
        copyTokenMajorRange(existing, grown, currentLength, numHeads, headDim);
        existing.close();
        return grown;
    }

    static AccelTensor growHeadMajor(AccelTensor existing, int newCapacity, int numHeads, int headDim,
            int currentLength) {
        long batch = existing.size(0);
        AccelTensor grown = AccelTensor.zeros(batch, numHeads, newCapacity, headDim);
        copyHeadMajorRange(existing, grown, currentLength, numHeads, headDim);
        existing.close();
        return grown;
    }

    static void appendTokenMajor(AccelTensor destination, AccelTensor delta, int tokenOffset,
            int numHeads, int headDim) {
        long batch = destination.size(0);
        int deltaTokens = Math.toIntExact(delta.size(1));
        long perTokenBytes = (long) numHeads * headDim * Float.BYTES;
        long srcBatchStrideBytes = (long) deltaTokens * perTokenBytes;
        long dstBatchStrideBytes = destination.size(1) * perTokenBytes;
        MemorySegment src = delta.dataPtr();
        MemorySegment dst = destination.dataPtr();
        for (int b = 0; b < batch; b++) {
            long srcOffset = b * srcBatchStrideBytes;
            long dstOffset = b * dstBatchStrideBytes + (long) tokenOffset * perTokenBytes;
            MemorySegment.copy(src, srcOffset, dst, dstOffset, srcBatchStrideBytes);
        }
    }

    static void appendTokenMajorToHeadMajor(AccelTensor destination, AccelTensor delta, int tokenOffset,
            int numHeads, int headDim) {
        long batch = destination.size(0);
        int deltaTokens = Math.toIntExact(delta.size(1));
        MemorySegment src = delta.dataPtr();
        MemorySegment dst = destination.dataPtr();
        long deltaBatchStrideBytes = (long) deltaTokens * numHeads * headDim * Float.BYTES;
        long deltaTokenStrideBytes = (long) numHeads * headDim * Float.BYTES;
        long deltaHeadStrideBytes = (long) headDim * Float.BYTES;
        long dstBatchStrideBytes = (long) numHeads * destination.size(2) * headDim * Float.BYTES;
        long dstHeadStrideBytes = destination.size(2) * (long) headDim * Float.BYTES;
        long copyBytes = (long) headDim * Float.BYTES;
        for (int b = 0; b < batch; b++) {
            long srcBatchBase = b * deltaBatchStrideBytes;
            long dstBatchBase = b * dstBatchStrideBytes;
            for (int tok = 0; tok < deltaTokens; tok++) {
                long srcTokenBase = srcBatchBase + tok * deltaTokenStrideBytes;
                long dstTokenBase = dstBatchBase + (long) (tokenOffset + tok) * headDim * Float.BYTES;
                for (int h = 0; h < numHeads; h++) {
                    long srcOffset = srcTokenBase + h * deltaHeadStrideBytes;
                    long dstOffset = dstTokenBase + h * dstHeadStrideBytes;
                    MemorySegment.copy(src, srcOffset, dst, dstOffset, copyBytes);
                }
            }
        }
    }

    static void packTokenMajorToHeadMajor(AccelTensor source, AccelTensor destination, int tokenCount,
            int numHeads, int headDim) {
        if (tokenCount <= 0) {
            return;
        }
        long batch = source.size(0);
        MemorySegment src = source.dataPtr();
        MemorySegment dst = destination.dataPtr();
        long srcBatchStrideBytes = source.size(1) * (long) numHeads * headDim * Float.BYTES;
        long srcTokenStrideBytes = (long) numHeads * headDim * Float.BYTES;
        long srcHeadStrideBytes = (long) headDim * Float.BYTES;
        long dstBatchStrideBytes = (long) numHeads * destination.size(2) * headDim * Float.BYTES;
        long dstHeadStrideBytes = destination.size(2) * (long) headDim * Float.BYTES;
        long copyBytes = (long) headDim * Float.BYTES;
        for (int b = 0; b < batch; b++) {
            long srcBatchBase = b * srcBatchStrideBytes;
            long dstBatchBase = b * dstBatchStrideBytes;
            for (int tok = 0; tok < tokenCount; tok++) {
                long srcTokenBase = srcBatchBase + tok * srcTokenStrideBytes;
                long dstTokenBase = dstBatchBase + (long) tok * headDim * Float.BYTES;
                for (int h = 0; h < numHeads; h++) {
                    long srcOffset = srcTokenBase + h * srcHeadStrideBytes;
                    long dstOffset = dstTokenBase + h * dstHeadStrideBytes;
                    MemorySegment.copy(src, srcOffset, dst, dstOffset, copyBytes);
                }
            }
        }
    }

    private static void copyTokenMajorRange(AccelTensor source, AccelTensor destination, int tokenCount,
            int numHeads, int headDim) {
        if (tokenCount <= 0) {
            return;
        }
        long batch = source.size(0);
        long copyBytesPerBatch = (long) tokenCount * numHeads * headDim * Float.BYTES;
        long srcBatchStrideBytes = source.size(1) * (long) numHeads * headDim * Float.BYTES;
        long dstBatchStrideBytes = destination.size(1) * (long) numHeads * headDim * Float.BYTES;
        MemorySegment src = source.dataPtr();
        MemorySegment dst = destination.dataPtr();
        for (int b = 0; b < batch; b++) {
            MemorySegment.copy(src, b * srcBatchStrideBytes, dst, b * dstBatchStrideBytes, copyBytesPerBatch);
        }
    }

    private static void copyHeadMajorRange(AccelTensor source, AccelTensor destination, int tokenCount,
            int numHeads, int headDim) {
        if (tokenCount <= 0) {
            return;
        }
        long batch = source.size(0);
        long copyBytesPerHead = (long) tokenCount * headDim * Float.BYTES;
        long srcBatchStrideBytes = (long) numHeads * source.size(2) * headDim * Float.BYTES;
        long dstBatchStrideBytes = (long) numHeads * destination.size(2) * headDim * Float.BYTES;
        long srcHeadStrideBytes = source.size(2) * (long) headDim * Float.BYTES;
        long dstHeadStrideBytes = destination.size(2) * (long) headDim * Float.BYTES;
        MemorySegment src = source.dataPtr();
        MemorySegment dst = destination.dataPtr();
        for (int b = 0; b < batch; b++) {
            long srcBatchBase = b * srcBatchStrideBytes;
            long dstBatchBase = b * dstBatchStrideBytes;
            for (int h = 0; h < numHeads; h++) {
                MemorySegment.copy(src, srcBatchBase + h * srcHeadStrideBytes,
                        dst, dstBatchBase + h * dstHeadStrideBytes, copyBytesPerHead);
            }
        }
    }
}
