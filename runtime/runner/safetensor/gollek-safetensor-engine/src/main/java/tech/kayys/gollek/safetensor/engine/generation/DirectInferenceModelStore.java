/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import static tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler.runWithProfileSuspended;

import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;
import tech.kayys.gollek.safetensor.engine.warmup.ModelWarmupService;
import tech.kayys.alkhawarizm.safetensor.quantization.QuantizationEngine;
import tech.kayys.alkhawarizm.safetensor.spi.SafetensorEngine;

import java.nio.file.Path;
import java.util.Collection;
import java.util.function.Supplier;

/**
 * Owns direct model lifecycle, warmup, release, and lookup behavior.
 */
final class DirectInferenceModelStore {
    private final Supplier<DirectModelLoader> modelLoader;
    private final Supplier<DirectForwardPass> forwardPass;
    private final Supplier<ModelWarmupService> modelWarmupService;
    private final Logger log;
    private DirectModelLifecycle<DirectInferenceEngine.LoadedModel> modelLifecycle;

    DirectInferenceModelStore(Supplier<DirectModelLoader> modelLoader,
            Supplier<DirectForwardPass> forwardPass,
            Supplier<ModelWarmupService> modelWarmupService,
            Logger log) {
        this.modelLoader = modelLoader;
        this.forwardPass = forwardPass;
        this.modelWarmupService = modelWarmupService;
        this.log = log;
    }

    String load(Path modelPath, QuantizationEngine.QuantStrategy quantStrategy) {
        return modelLifecycle().load(modelPath, quantStrategy);
    }

    void unload(Path modelPath) {
        modelLifecycle().unload(modelPath);
    }

    SafetensorEngine.LoadedModel find(Path modelPath) {
        return modelLifecycle().find(modelPath);
    }

    SafetensorEngine.LoadedModel findByKey(String key) {
        return modelLifecycle().findByKey(key);
    }

    boolean contains(Path modelPath) {
        return modelLifecycle().contains(modelPath);
    }

    Collection<DirectInferenceEngine.LoadedModel> snapshot() {
        return modelLifecycle().snapshot();
    }

    DirectInferenceEngine.LoadedModel require(Path modelPath, boolean verbose, String debugPrefix) {
        return modelLifecycle().require(modelPath, verbose, debugPrefix);
    }

    private DirectModelLifecycle<DirectInferenceEngine.LoadedModel> modelLifecycle() {
        if (modelLifecycle == null) {
            modelLifecycle = new DirectModelLifecycle<>(
                    (path, strategy) -> modelLoader.get().load(path, strategy),
                    DirectInferenceEngine.LoadedModel::getQuantStrategy,
                    this::warmUpModel,
                    this::releaseModel,
                    log);
        }
        return modelLifecycle;
    }

    private void warmUpModel(DirectInferenceEngine.LoadedModel model) {
        ModelWarmupService warmup = modelWarmupService == null ? null : modelWarmupService.get();
        if (warmup != null) {
            runWithProfileSuspended(() -> warmup.warmUp(model));
        }
    }

    private void releaseModel(DirectInferenceEngine.LoadedModel model) {
        DirectLoadedModelReleaser.release(model,
                weights -> forwardPass.get().clearResolvedModelWeights(weights));
    }
}
