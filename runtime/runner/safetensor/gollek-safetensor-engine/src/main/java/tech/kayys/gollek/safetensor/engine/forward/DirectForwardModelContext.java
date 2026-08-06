/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import jakarta.inject.Singleton;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.Map;

/**
 * Owns per-model forward traits and resolved-weight caching.
 */
@Singleton
public class DirectForwardModelContext {
    private final DirectForwardWeightResolver weightResolver = new DirectForwardWeightResolver();
    private volatile ModelConfigTraits lastModelConfigTraits = ModelConfigTraits.EMPTY;

    void clearResolvedWeights(Map<String, AccelTensor> weights) {
        weightResolver.clear(weights);
    }

    ResolvedModelWeights resolveWeights(Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch) {
        boolean addOneRmsNorm = useAddOneRmsNorm(arch, config);
        return weightResolver.resolve(weights, config, arch, addOneRmsNorm);
    }

    ModelConfigTraits traits(ModelConfig config) {
        return traits(config, null);
    }

    ModelConfigTraits traits(ModelConfig config, ModelArchitecture arch) {
        if (config == null) {
            return ModelConfigTraits.EMPTY;
        }
        ModelConfigTraits cached = lastModelConfigTraits;
        if (cached.matches(config)) {
            return cached;
        }
        ModelConfigTraits traits = ModelConfigTraits.create(config, arch);
        lastModelConfigTraits = traits;
        return traits;
    }

    private boolean useAddOneRmsNorm(ModelArchitecture arch, ModelConfig config) {
        ModelConfigTraits traits = traits(config, arch);
        if (traits.geluGatedFfn()) {
            // GeGLU-gated models (Gemma 3, Gemma 4, etc.) use output * (1 + weight) RMSNorm.
            return true;
        }
        if (arch != null) {
            return arch.addOneToRmsNormWeight();
        }
        return false;
    }
}
