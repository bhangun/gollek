/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import java.util.Random;

final class TokenSamplingProbabilityMass {
    private TokenSamplingProbabilityMass() {
    }

    static int build(
            float[] logits,
            int[] indices,
            double[] probs,
            int limit,
            float temp,
            float topP,
            float minP,
            float softCap) {
        float maxLogit = logits[indices[0]];
        float normalizationMaxLogit = softCap > 0
                ? applySoftCap(maxLogit, softCap)
                : maxLogit;
        double minProbLimit = minP > 0.0f
                ? minP * Math.exp((maxLogit - maxLogit) / temp)
                : 0.0;
        double sum = 0.0;
        int actualElements = 0;

        for (int i = 0; i < limit; i++) {
            int idx = indices[i];
            float logit = logits[idx];
            if (softCap > 0) {
                logit = applySoftCap(logit, softCap);
            }
            double p = Math.exp((logit - normalizationMaxLogit) / temp);

            if (minP > 0.0f && p < minProbLimit && actualElements > 0) {
                break;
            }

            probs[i] = p;
            sum += p;
            actualElements++;
        }

        return applyTopP(probs, actualElements, sum, topP);
    }

    static int draw(int[] indices, double[] probs, int actualElements, Random rng) {
        double filteredSum = 0.0;
        for (int i = 0; i < actualElements; i++) {
            filteredSum += probs[i];
        }

        double r = rng.nextDouble() * filteredSum;
        double cumulative = 0.0;
        for (int i = 0; i < actualElements; i++) {
            cumulative += probs[i];
            if (cumulative >= r) {
                return indices[i];
            }
        }

        return indices[0];
    }

    private static float applySoftCap(float logit, float softCap) {
        return (float) (softCap * Math.tanh(logit / softCap));
    }

    private static int applyTopP(double[] probs, int actualElements, double sum, float topP) {
        if (topP <= 0.0f || topP >= 1.0f) {
            return actualElements;
        }

        double cumulative = 0.0;
        for (int i = 0; i < actualElements; i++) {
            double pNorm = probs[i] / sum;
            cumulative += pNorm;
            if (cumulative > topP && i > 0) {
                return i + 1;
            }
        }
        return actualElements;
    }
}
