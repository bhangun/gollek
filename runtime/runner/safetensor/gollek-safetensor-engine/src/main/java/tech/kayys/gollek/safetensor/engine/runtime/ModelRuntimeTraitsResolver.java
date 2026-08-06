/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.runtime;

import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

/**
 * Resolves the effective runtime policy for SafeTensor execution.
 *
 * <p>Model farms should provide the primary traits through their architecture
 * adapters. Config-derived traits are only a compatibility fallback.</p>
 */
public final class ModelRuntimeTraitsResolver {

    private ModelRuntimeTraitsResolver() {
    }

    public static ModelRuntimeTraits resolve(ModelConfig config) {
        return resolve(null, config, null);
    }

    public static ModelRuntimeTraits resolve(ModelConfig config, ModelRuntimeTraits providedTraits) {
        return resolve(null, config, providedTraits);
    }

    public static ModelRuntimeTraits resolve(ModelArchitecture architecture, ModelConfig config) {
        return resolve(architecture, config, null);
    }

    public static ModelRuntimeTraits resolve(ModelArchitecture architecture, ModelConfig config,
            ModelRuntimeTraits providedTraits) {
        ModelRuntimeTraits traits = providedTraits;
        if (traits == null && architecture != null) {
            traits = architecture.runtimeTraits(config);
        }
        if (traits == null) {
            traits = ModelRuntimeTraits.fallbackFromConfig(config);
        }
        return traits.withDetectedModalities(config);
    }
}
