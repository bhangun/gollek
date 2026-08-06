/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TokenSamplingProbabilityMassTest {

    @Test
    void buildStopsAtMinPThresholdAfterKeepingBestCandidate() {
        float[] logits = { 10.0f, 8.0f, 7.0f };
        int[] indices = { 0, 1, 2 };
        double[] probs = new double[3];

        int actual = TokenSamplingProbabilityMass.build(logits, indices, probs, 3, 1.0f, 1.0f, 0.2f, 0.0f);

        assertEquals(1, actual);
        assertEquals(1.0, probs[0], 0.0001);
    }

    @Test
    void buildAppliesTopPAfterProbabilityMassIsBuilt() {
        float[] logits = { 10.0f, 9.0f, 1.0f };
        int[] indices = { 0, 1, 2 };
        double[] probs = new double[3];

        int actual = TokenSamplingProbabilityMass.build(logits, indices, probs, 3, 1.0f, 0.8f, 0.0f, 0.0f);

        assertEquals(2, actual);
        assertTrue(probs[0] > probs[1]);
        assertTrue(probs[1] > probs[2]);
    }

    @Test
    void drawUsesFilteredProbabilityMassOnly() {
        int[] indices = { 4, 8, 12 };
        double[] probs = { 0.1, 0.9, 100.0 };

        int token = TokenSamplingProbabilityMass.draw(indices, probs, 2, fixedRandom(0.99));

        assertEquals(8, token);
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
