/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.mask;

import tech.kayys.gollek.spi.model.ModelConfig;
import java.util.Arrays;

/**
 * Sliding window attention (SWA) mask implementation.
 * Used by Mistral and Mixtral models to limit attention context.
 */
public class SlidingWindowAttention {

    /**
     * Check if sliding window attention is enabled in the model config.
     */
    public static boolean isEnabled(ModelConfig config) {
        return config.hasSlidingWindow();
    }

    /**
     * Get the window size from model config.
     */
    public static int windowSize(ModelConfig config) {
        return config.slidingWindowSize();
    }

    /**
     * Build a sliding window causal mask.
     *
     * @param sq seqLen query
     * @param sk seqLen key
     * @param offset context offset
     * @param windowSize window size
     * @return float array with mask values
     */
    public static float[] buildMask(int sq, int sk, int offset, int windowSize) {
        float[] mask = new float[sq * sk];
        Arrays.fill(mask, -1e10f);

        for (int i = 0; i < sq; i++) {
            int absPos = offset + i;
            // Token i can attend to tokens in [absPos - windowSize + 1, absPos]
            int minJ = Math.max(0, absPos - windowSize + 1);
            int maxJ = absPos;
            
            for (int j = minJ; j <= maxJ && j < sk; j++) {
                mask[i * sk + j] = 0f;
            }
        }
        return mask;
    }
}
