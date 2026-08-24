/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.Random;

final class TokenSamplingDistribution {

    private final ThreadLocal<SamplingWorkspace> workspaces = ThreadLocal.withInitial(SamplingWorkspace::new);

    int sample(float[] logits, GenerationConfig config, ModelConfig modelConfig, Random rng) {
        float temp = config.temperature();
        if (temp <= 0) {
            temp = 1.0f;
        }

        SamplingWorkspace workspace = workspaces.get();
        TokenSamplingCandidates.Prepared candidates = TokenSamplingCandidates.prepare(
                logits,
                config.topK(),
                workspace.indices(logits.length));

        double[] probs = workspace.probs(candidates.limit());
        int actualElements = TokenSamplingProbabilityMass.build(
                logits,
                candidates.indices(),
                probs,
                candidates.limit(),
                temp,
                config.topP(),
                config.minP(),
                finalLogitSoftCap(modelConfig));

        return TokenSamplingProbabilityMass.draw(candidates.indices(), probs, actualElements, rng);
    }

    private static float finalLogitSoftCap(ModelConfig modelConfig) {
        Double cap = modelConfig == null ? null : modelConfig.finalLogitSoftcapping();
        return cap != null && cap > 0 ? cap.floatValue() : 0.0f;
    }

    private static final class SamplingWorkspace {
        private int[] indices;
        private double[] probs;

        int[] indices(int size) {
            if (indices == null || indices.length < size) {
                indices = new int[size];
            }
            return indices;
        }

        double[] probs(int size) {
            if (probs == null || probs.length < size) {
                probs = new double[size];
            }
            return probs;
        }
    }
}
