/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.gollek.safetensor.generation.GenerationConfig;

final class TokenSamplingPenaltyPolicy {
    private TokenSamplingPenaltyPolicy() {
    }

    static void apply(float[] logits, int[] freq, GenerationConfig config) {
        if (logits == null || freq == null || config == null) {
            return;
        }

        float repetitionPenalty = config.repetitionPenalty();
        float frequencyPenalty = config.frequencyPenalty();
        if (repetitionPenalty > 1.0f || frequencyPenalty > 0.0f) {
            applyFrequencyPenalties(logits, freq, repetitionPenalty, frequencyPenalty);
        }

        // Deterministic decoding is especially vulnerable to prompt-echo loops.
        // Once a prompt token has also been generated, remove it from greedy reuse.
        if (config.requestsGreedyDecoding() && repetitionPenalty > 1.0f) {
            suppressRepeatedGreedyTokens(logits, freq);
        }
    }

    private static void applyFrequencyPenalties(
            float[] logits,
            int[] freq,
            float repetitionPenalty,
            float frequencyPenalty) {
        for (int i = 0; i < logits.length && i < freq.length; i++) {
            if (freq[i] <= 0) {
                continue;
            }
            if (repetitionPenalty > 1.0f) {
                float effectiveRepPenalty = freq[i] > 1
                        ? (float) Math.pow(repetitionPenalty, freq[i])
                        : repetitionPenalty;
                if (logits[i] > 0) {
                    logits[i] /= effectiveRepPenalty;
                } else {
                    logits[i] *= effectiveRepPenalty;
                }
            }
            if (frequencyPenalty > 0.0f) {
                logits[i] -= frequencyPenalty * freq[i];
            }
        }
    }

    private static void suppressRepeatedGreedyTokens(float[] logits, int[] freq) {
        for (int i = 0; i < logits.length && i < freq.length; i++) {
            if (freq[i] > 1) {
                logits[i] = Float.NEGATIVE_INFINITY;
            }
        }
    }
}
