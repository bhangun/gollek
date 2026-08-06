/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TokenSamplingDistributionTest {

    private final TokenSamplingDistribution distribution = new TokenSamplingDistribution();

    @Test
    void samplesSingleCandidateWithoutSortingStackGrowth() {
        float[] logits = { 3.0f };
        GenerationConfig config = GenerationConfig.builder()
                .strategy(GenerationConfig.SamplingStrategy.TOP_K)
                .topK(0)
                .build();

        int token = distribution.sample(logits, config, null, fixedRandom(0.99));

        assertEquals(0, token);
    }

    @Test
    void samplesFromTopKCandidatesOnly() {
        float[] logits = { 10.0f, 9.0f, 1.0f };
        GenerationConfig config = GenerationConfig.builder()
                .strategy(GenerationConfig.SamplingStrategy.TOP_K)
                .topK(2)
                .topP(1.0f)
                .build();

        int token = distribution.sample(logits, config, null, fixedRandom(0.99));

        assertEquals(1, token);
    }

    @Test
    void minPStopsAfterBestTokenWhenNextCandidateFallsBelowThreshold() {
        float[] logits = { 10.0f, 8.0f, 7.0f };
        GenerationConfig config = GenerationConfig.builder()
                .strategy(GenerationConfig.SamplingStrategy.TOP_K)
                .topK(3)
                .minP(0.2f)
                .build();

        int token = distribution.sample(logits, config, null, fixedRandom(0.99));

        assertEquals(0, token);
    }

    @Test
    void topPFiltersTailBeforeSampling() {
        float[] logits = { 10.0f, 9.0f, 1.0f };
        GenerationConfig config = GenerationConfig.builder()
                .strategy(GenerationConfig.SamplingStrategy.TOP_K)
                .topK(3)
                .topP(0.8f)
                .build();

        int token = distribution.sample(logits, config, null, fixedRandom(0.99));

        assertEquals(1, token);
    }

    private static Random fixedRandom(double value) {
        return new Random(0L) {
            @Override
            public double nextDouble() {
                return value;
            }
        };
    }
}
