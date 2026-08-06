/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import java.util.Arrays;

final class AttentionOnlineSoftmax {
    private static final float NORMALIZER_EPSILON = 1e-9f;

    private final float[] accumulator;
    private final int headDim;
    private float maxScore;
    private float normalizer;
    private float previousWeight;
    private float currentWeight;

    AttentionOnlineSoftmax(float[] accumulator, int headDim) {
        if (headDim < 0 || accumulator.length < headDim) {
            throw new IllegalArgumentException(
                    "Accumulator length " + accumulator.length + " is smaller than headDim " + headDim);
        }
        this.accumulator = accumulator;
        this.headDim = headDim;
        reset();
    }

    void reset() {
        maxScore = Float.NEGATIVE_INFINITY;
        normalizer = 0.0f;
        previousWeight = 0.0f;
        currentWeight = 0.0f;
        Arrays.fill(accumulator, 0, headDim, 0.0f);
    }

    void observe(float score) {
        float previousMaxScore = maxScore;
        maxScore = Math.max(maxScore, score);
        previousWeight = (float) Math.exp(previousMaxScore - maxScore);
        currentWeight = (float) Math.exp(score - maxScore);
        normalizer = normalizer * previousWeight + currentWeight;
    }

    float previousWeight() {
        return previousWeight;
    }

    float currentWeight() {
        return currentWeight;
    }

    float inverseNormalizer() {
        return 1.0f / (normalizer + NORMALIZER_EPSILON);
    }

    float[] accumulator() {
        return accumulator;
    }
}
