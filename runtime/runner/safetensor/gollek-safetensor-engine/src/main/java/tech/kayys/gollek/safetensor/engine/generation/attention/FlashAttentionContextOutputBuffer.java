/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import java.lang.foreign.MemorySegment;

/**
 * Creates attention-context output tensors backed by caller-owned scratch
 * memory, falling back to normal tensor allocation when the scratch is absent.
 */
final class FlashAttentionContextOutputBuffer {
    private FlashAttentionContextOutputBuffer() {
    }

    static AccelTensor viewOrAllocate(MemorySegment buffer, AccelTensor query) {
        AccelTensor view = view(buffer, query);
        return view == null ? AccelTensor.zeros(query.shape()) : view;
    }

    static AccelTensor view(MemorySegment buffer, AccelTensor query) {
        if (buffer == null || query == null) {
            return null;
        }
        long requiredBytes = Math.multiplyExact(query.numel(), (long) Float.BYTES);
        if (buffer.byteSize() < requiredBytes) {
            return null;
        }
        if (overlaps(buffer, requiredBytes, query.dataPtr(), requiredBytes)) {
            return null;
        }
        return AccelTensor.view(buffer.asSlice(0, requiredBytes), query.shape());
    }

    private static boolean overlaps(MemorySegment first, long firstBytes, MemorySegment second, long secondBytes) {
        long firstStart = first.address();
        long secondStart = second.address();
        long firstEnd = Math.addExact(firstStart, firstBytes);
        long secondEnd = Math.addExact(secondStart, secondBytes);
        return firstStart < secondEnd && secondStart < firstEnd;
    }
}
