/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.mask;

import java.util.Arrays;

/**
 * Causal mask kernel for attention.
 *
 * <p>
 * Generates a lower-triangular mask (1s on and below diagonal, 0s above)
 * and provides utility to apply it to attention scores as -inf.
 */
public class CausalMaskKernel {

    /**
     * Build a causal mask for a score matrix of shape [sq, sk].
     *
     * @param sq seqLen query
     * @param sk seqLen key (context)
     * @param offset starting position in context
     * @return float array of size sq*sk containing 0 for kept, -1e10 for masked
     */
    public static float[] buildCausalMask(int sq, int sk, int offset) {
        float[] mask = new float[sq * sk];
        Arrays.fill(mask, -1e10f); // Default to masked (-inf)

        for (int i = 0; i < sq; i++) {
            // Context index must be <= absolute position of current token
            // Absolute position = offset + i
            int maxJ = offset + i;
            for (int j = 0; j <= maxJ && j < sk; j++) {
                mask[i * sk + j] = 0f; // Kept
            }
        }
        return mask;
    }

    /**
     * Apply a mask to scores in-place.
     */
    public static void addMask(float[] scores, float[] mask, int batch, int heads, int sq, int sk) {
        int matrixSize = sq * sk;
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < heads; h++) {
                int base = (b * heads + h) * matrixSize;
                for (int i = 0; i < matrixSize; i++) {
                    scores[base + i] += mask[i];
                }
            }
        }
    }
}
