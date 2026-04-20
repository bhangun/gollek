/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.mask;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Causal mask kernel for attention.
 */
public class CausalMaskKernel {

    /**
     * Apply a causal mask to scores in-place on a MemorySegment.
     *
     * @param seg the scores segment [batch, heads, sq, sk]
     * @param batch number of batches
     * @param heads number of heads
     * @param sq seqLen query
     * @param sk seqLen key
     * @param offset starting position in context
     */
    public static void applyCausalMask(MemorySegment seg, int batch, int heads, int rows, int cols, int startPos) {
        // Use a much larger penalty for zero-softmasking (-1e18 is safer than -1e10 for large scores)
        final float MASK_VALUE = -1e18f;
        
        // Loop order optimized for segment layout: [batch, heads, rows, cols]
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < heads; h++) {
                for (int row = 0; row < rows; row++) {
                    int queryPos = startPos + row;
                    // Future tokens in the cache (col > queryPos) must be masked.
                    for (int col = queryPos + 1; col < cols; col++) {
                        long index = (((long)b * heads + h) * rows + row) * cols + col;
                        seg.setAtIndex(ValueLayout.JAVA_FLOAT, index, MASK_VALUE);
                    }
                }
            }
        }
    }
}
