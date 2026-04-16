/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SlidingWindowAttention.java
 * ────────────────────────────
 * Applies the sliding window attention constraint used by Mistral 7B, Mixtral,
 * Mistral-NeMo, and other models with a finite attention context window.
 *
 * What sliding window attention is
 * ══════════════════════════════════
 * Standard causal attention allows token i to attend to ALL tokens 0..i.
 * For long contexts this means the KV cache grows linearly with sequence length.
 *
 * Sliding window attention restricts token i to attend only to:
 *   max(0, i - windowSize) .. i
 *
 * For Mistral 7B, windowSize = 4096.
 * Even though the model has a 32K context limit, each token only has direct
 * attention to the previous 4096 tokens. Information from further back
 * propagates through multiple layers (the "sliding window" effect across layers).
 *
 * Implementation
 * ══════════════
 * This is a two-line change to CausalMaskKernel.buildCausalMask:
 *   Original:  maxKV = startPos + qi
 *   With SWA:  maxKV = min(startPos + qi, startPos + qi - seqKV + windowSize + 1)
 *
 * More precisely, for query position absolutePos = startPos + qi:
 *   allowed kv range: [absolutePos - windowSize, absolutePos]
 *   minKV = max(0, absolutePos - windowSize)
 *   maxKV = absolutePos
 *
 * We add a windowStart per query to the mask — any kv position < windowStart
 * for that query gets -inf regardless of causal constraint.
 *
 * FlashAttentionKernel integration
 * ══════════════════════════════════
 * FlashAttentionKernel.tiledAttention() calls:
 *   int[] maskParams = SlidingWindowAttention.getMaskParams(config, startPos, seqQ, seqKV);
 *   mask = CausalMaskKernel.buildMask(seqQ, seqKV, startPos, maskParams[0]); // windowSize
 *
 * When windowSize == Integer.MAX_VALUE (no SWA), the mask is identical to
 * standard causal masking.
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.spi.model.ModelConfig;

/**
 * Sliding window attention mask utilities.
 *
 * <p>
 * For Mistral/Mixtral models, limits attention range to the last
 * {@code windowSize} tokens per query position.
 */
public final class SlidingWindowAttention {

    private SlidingWindowAttention() {
    }

    /**
     * Build a causal mask that also applies a sliding window constraint.
     *
     * <p>
     * For each query position {@code qi} (0-based within the current block):
     * <ul>
     * <li>Allow keys in {@code [max(0, absPos - windowSize), absPos]}.
     * <li>All other key positions get {@code -inf}.
     * </ul>
     *
     * <p>
     * When {@code windowSize == Integer.MAX_VALUE} this is identical to
     * a standard causal mask.
     *
     * @param seqQ       number of query positions in this block
     * @param seqKV      total number of key-value positions in the cache
     * @param startPos   absolute position of the first query token
     * @param windowSize sliding window size (from ModelConfig, or MAX_VALUE for
     *                   full attention)
     * @return flat float[] of shape [seqQ × seqKV] with 0.0 for allowed and -inf
     *         for masked
     */
    public static float[] buildMask(int seqQ, int seqKV, int startPos, int windowSize) {
        float[] mask = new float[seqQ * seqKV];
        for (int qi = 0; qi < seqQ; qi++) {
            int absPos = startPos + qi;
            int minKV = windowSize == Integer.MAX_VALUE ? 0
                    : Math.max(0, absPos - windowSize + 1);
            int maxKV = absPos;
            for (int kj = 0; kj < seqKV; kj++) {
                mask[qi * seqKV + kj] = (kj >= minKV && kj <= maxKV)
                        ? 0f
                        : Float.NEGATIVE_INFINITY;
            }
        }
        return mask;
    }

    /**
     * Resolve the effective window size from a model config.
     *
     * <p>
     * Returns {@link Integer#MAX_VALUE} when no sliding window is configured
     * (standard full causal attention).
     *
     * @param config model configuration
     * @return window size, or Integer.MAX_VALUE for full attention
     */
    public static int windowSize(ModelConfig config) {
        return config.hasSlidingWindow()
                ? config.slidingWindowSize()
                : Integer.MAX_VALUE;
    }

    /**
     * Resolve window size for a specific layer (Gemma-4 sliding/full attention mix).
     */
    public static int windowSize(ModelConfig config, int layerIdx) {
        return config.isSlidingAttentionLayer(layerIdx)
                ? config.slidingWindowSize()
                : Integer.MAX_VALUE;
    }

    /**
     * Whether a given model uses sliding window attention.
     */
    public static boolean isEnabled(ModelConfig config) {
        return config.hasSlidingWindow();
    }

    /**
     * Whether sliding window applies to a specific layer.
     */
    public static boolean isEnabled(ModelConfig config, int layerIdx) {
        return config.isSlidingAttentionLayer(layerIdx);
    }

    /**
     * For the decode step (seqQ=1), check whether the single query token
     * is outside the window of any cached key. If the absolute position
     * exceeds windowSize, the earliest KV blocks can be evicted.
     *
     * @param absPos absolute position of the token being decoded
     * @param config model configuration
     * @return the earliest key position that is still within the window
     */
    public static int effectiveKVStart(int absPos, ModelConfig config) {
        int ws = windowSize(config);
        if (ws == Integer.MAX_VALUE)
            return 0;
        return Math.max(0, absPos - ws + 1);
    }
}
