/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class AttentionOnlineSoftmaxTest {

    @Test
    void tracksOnlineSoftmaxWeightsAndInverseNormalizer() {
        AttentionOnlineSoftmax softmax = new AttentionOnlineSoftmax(new float[2], 2);

        softmax.observe(2.0f);

        assertEquals(0.0f, softmax.previousWeight(), 0.0001f);
        assertEquals(1.0f, softmax.currentWeight(), 0.0001f);
        assertEquals(1.0f, softmax.inverseNormalizer(), 0.0001f);

        softmax.observe(4.0f);

        float previousWeight = (float) Math.exp(-2.0f);
        assertEquals(previousWeight, softmax.previousWeight(), 0.0001f);
        assertEquals(1.0f, softmax.currentWeight(), 0.0001f);
        assertEquals(1.0f / (previousWeight + 1.0f), softmax.inverseNormalizer(), 0.0001f);
    }

    @Test
    void resetClearsActiveAccumulatorAndWeights() {
        float[] accumulator = { 7.0f, 8.0f, 9.0f };
        AttentionOnlineSoftmax softmax = new AttentionOnlineSoftmax(accumulator, 2);
        accumulator[0] = 5.0f;
        accumulator[1] = 6.0f;
        softmax.observe(1.0f);

        softmax.reset();

        assertArrayEquals(new float[] { 0.0f, 0.0f, 9.0f }, accumulator, 0.0001f);
        assertEquals(0.0f, softmax.previousWeight(), 0.0001f);
        assertEquals(0.0f, softmax.currentWeight(), 0.0001f);
    }

    @Test
    void rejectsAccumulatorSmallerThanHeadDim() {
        assertThrows(IllegalArgumentException.class, () -> new AttentionOnlineSoftmax(new float[1], 2));
    }
}
