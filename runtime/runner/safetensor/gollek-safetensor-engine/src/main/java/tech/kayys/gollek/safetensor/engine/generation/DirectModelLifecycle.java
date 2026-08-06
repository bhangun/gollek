/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.safetensor.quantization.QuantizationEngine;
import tech.kayys.alkhawarizm.safetensor.spi.SafetensorEngine;

import java.nio.file.Path;
import java.util.Collection;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Owns loaded-model registry state and lifecycle orchestration for direct
 * SafeTensor generation.
 */
final class DirectModelLifecycle<T extends SafetensorEngine.LoadedModel> {
    private final DirectLoadedModelRegistry<T> loadedModels = new DirectLoadedModelRegistry<>();
    private final BiFunction<Path, QuantizationEngine.QuantStrategy, T> modelLoader;
    private final Function<T, QuantizationEngine.QuantStrategy> quantStrategy;
    private final Consumer<T> modelWarmup;
    private final Consumer<T> modelReleaser;
    private final Logger log;

    DirectModelLifecycle(
            BiFunction<Path, QuantizationEngine.QuantStrategy, T> modelLoader,
            Function<T, QuantizationEngine.QuantStrategy> quantStrategy,
            Consumer<T> modelWarmup,
            Consumer<T> modelReleaser,
            Logger log) {
        this.modelLoader = Objects.requireNonNull(modelLoader, "modelLoader");
        this.quantStrategy = Objects.requireNonNull(quantStrategy, "quantStrategy");
        this.modelWarmup = modelWarmup == null ? ignored -> {
        } : modelWarmup;
        this.modelReleaser = modelReleaser == null ? ignored -> {
        } : modelReleaser;
        this.log = log == null ? Logger.getLogger(DirectModelLifecycle.class) : log;
    }

    String load(Path modelPath, QuantizationEngine.QuantStrategy requestedQuantStrategy) {
        QuantizationEngine.QuantStrategy effectiveQuantStrategy =
                DirectModelLoader.normalizeQuantStrategy(requestedQuantStrategy);
        Path resolved = loadedModels.resolve(modelPath);

        T existing = loadedModels.find(resolved);
        if (usesQuantStrategy(existing, effectiveQuantStrategy)) {
            log.infof("DirectInferenceEngine: model already loaded [%s] (strategy=%s)",
                    resolved.getFileName(), effectiveQuantStrategy);
            return existing.key();
        }

        synchronized (loadedModels.lockFor(resolved)) {
            existing = loadedModels.find(resolved);
            if (usesQuantStrategy(existing, effectiveQuantStrategy)) {
                return existing.key();
            }
            if (existing != null) {
                log.infof("DirectInferenceEngine: reloading [%s] for quantization strategy change %s -> %s",
                        resolved.getFileName(), normalizedQuantStrategy(existing), effectiveQuantStrategy);
                unload(resolved);
            }

            log.infof("DirectInferenceEngine: loading model [%s] (strategy=%s)",
                    resolved.getFileName(), effectiveQuantStrategy);

            T model = modelLoader.apply(resolved, effectiveQuantStrategy);
            loadedModels.register(resolved, model);
            modelWarmup.accept(model);

            log.infof("DirectInferenceEngine: loaded [%s] — %d weights, arch=%s",
                    model.key(), model.weights().size(), model.config().getModelType());
            return model.key();
        }
    }

    void unload(Path modelPath) {
        Path resolved = loadedModels.resolve(modelPath);
        T model = loadedModels.remove(resolved);
        if (model != null) {
            modelReleaser.accept(model);
            log.infof("DirectInferenceEngine: unloaded [%s]", resolved.getFileName());
        }
    }

    T require(Path modelPath, boolean verbose, String debugPrefix) {
        return DirectLoadedModelAcquirer.require(
                () -> find(modelPath),
                () -> load(modelPath, QuantizationEngine.QuantStrategy.NONE),
                DirectInferenceProfiler::recordModelLoadNanos,
                verbose,
                debugPrefix);
    }

    T find(Path modelPath) {
        return loadedModels.find(modelPath);
    }

    T findByKey(String key) {
        return loadedModels.findByKey(key);
    }

    boolean contains(Path modelPath) {
        return loadedModels.contains(modelPath);
    }

    Collection<T> snapshot() {
        return loadedModels.snapshot();
    }

    private boolean usesQuantStrategy(T model, QuantizationEngine.QuantStrategy expected) {
        return model != null && normalizedQuantStrategy(model) == expected;
    }

    private QuantizationEngine.QuantStrategy normalizedQuantStrategy(T model) {
        return DirectModelLoader.normalizeQuantStrategy(quantStrategy.apply(model));
    }
}
