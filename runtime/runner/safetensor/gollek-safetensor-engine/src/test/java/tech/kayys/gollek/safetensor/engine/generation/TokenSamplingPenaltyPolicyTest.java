/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TokenSamplingPenaltyPolicyTest {

    @Test
    void leavesLogitsUntouchedWithoutFrequenciesOrPenalties() {
        float[] logits = { 1.0f, -2.0f, 3.0f };

        TokenSamplingPenaltyPolicy.apply(logits, null, GenerationConfig.defaults());

        assertArrayEquals(new float[] { 1.0f, -2.0f, 3.0f }, logits);
    }

    @Test
    void appliesRepetitionPenaltyByFrequencyCount() {
        float[] logits = { 4.0f, -2.0f, 8.0f };
        int[] freq = { 1, 2, 0 };
        GenerationConfig config = GenerationConfig.builder()
                .strategy(GenerationConfig.SamplingStrategy.TOP_K)
                .repetitionPenalty(2.0f)
                .build();

        TokenSamplingPenaltyPolicy.apply(logits, freq, config);

        assertEquals(2.0f, logits[0], 0.0001f);
        assertEquals(-8.0f, logits[1], 0.0001f);
        assertEquals(8.0f, logits[2], 0.0001f);
    }

    @Test
    void appliesFrequencyPenaltyAfterRepetitionPenalty() {
        float[] logits = { 4.0f, 6.0f };
        int[] freq = { 2, 1 };
        GenerationConfig config = GenerationConfig.builder()
                .strategy(GenerationConfig.SamplingStrategy.TOP_K)
                .repetitionPenalty(2.0f)
                .frequencyPenalty(0.5f)
                .build();

        TokenSamplingPenaltyPolicy.apply(logits, freq, config);

        assertEquals(0.0f, logits[0], 0.0001f);
        assertEquals(2.5f, logits[1], 0.0001f);
    }

    @Test
    void greedyPromptEchoSuppressionRemovesRepeatedTokens() {
        float[] logits = { 9.0f, 8.0f, 7.0f };
        int[] freq = { 1, 2, 0 };
        GenerationConfig config = GenerationConfig.builder()
                .strategy(GenerationConfig.SamplingStrategy.GREEDY)
                .repetitionPenalty(1.2f)
                .build();

        TokenSamplingPenaltyPolicy.apply(logits, freq, config);

        assertEquals(7.5f, logits[0], 0.0001f);
        assertTrue(Float.isInfinite(logits[1]) && logits[1] < 0.0f);
        assertEquals(7.0f, logits[2], 0.0001f);
    }

    @Test
    void sampledModeKeepsRepeatedTokensAvailable() {
        float[] logits = { 9.0f, 8.0f };
        int[] freq = { 1, 2 };
        GenerationConfig config = GenerationConfig.builder()
                .strategy(GenerationConfig.SamplingStrategy.TOP_K)
                .repetitionPenalty(2.0f)
                .build();

        TokenSamplingPenaltyPolicy.apply(logits, freq, config);

        assertEquals(4.5f, logits[0], 0.0001f);
        assertEquals(2.0f, logits[1], 0.0001f);
    }
}
