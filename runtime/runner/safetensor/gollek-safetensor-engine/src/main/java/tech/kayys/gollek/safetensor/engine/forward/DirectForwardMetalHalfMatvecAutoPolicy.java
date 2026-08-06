/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.spi.model.ModelConfig;

record DirectForwardMetalHalfMatvecAutoPolicy(
        Boolean autoMetalHalfMatvec,
        int defaultMaxOutput) {

    static DirectForwardMetalHalfMatvecAutoPolicy from(DirectForwardMetalHalfMatvecCoreOptions options) {
        return new DirectForwardMetalHalfMatvecAutoPolicy(
                options.autoMetalHalfMatvec(),
                options.metalHalfMatvecMaxOutput());
    }

    boolean shouldUseMetalHalfMatvec(ModelConfigTraits traits, ModelConfig config, int outputDim) {
        return shouldUseMetalHalfMatvec(traits, config, outputDim, defaultMaxOutput);
    }

    boolean shouldUseMetalHalfMatvec(ModelConfigTraits traits, ModelConfig config, int outputDim, int maxOutput) {
        return autoMetalHalfMatvecEnabled()
                && maxOutput > 0
                && outputDim <= maxOutput
                && isMetalHalfMatvecAutoCandidate(traits, config);
    }

    private boolean autoMetalHalfMatvecEnabled() {
        if (autoMetalHalfMatvec != null) {
            return autoMetalHalfMatvec;
        }
        return true;
    }

    private static boolean isMetalHalfMatvecAutoCandidate(ModelConfigTraits traits, ModelConfig config) {
        if (traits.modelType().isBlank() || config == null) {
            return false;
        }
        if (traits.nativeBf16Matvec() || traits.perLayerInputEmbedding()) {
            return true;
        }
        if (traits.siluGated()) {
            return config.getNumHiddenLayers() >= 20
                    && config.getHiddenSize() <= 2048
                    && config.getIntermediateSize() >= 2048;
        }
        return config.getNumHiddenLayers() >= 30
                && config.getIntermediateSize() >= 4096
                && config.getHiddenSize() <= 4096;
    }
}
