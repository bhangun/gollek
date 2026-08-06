/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.alkhawarizm.safetensor.spi.SafetensorEngine;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Thread-safe registry for loaded direct SafeTensor models.
 */
final class DirectLoadedModelRegistry<T extends SafetensorEngine.LoadedModel> {
    private final Map<Path, T> modelsByPath = new ConcurrentHashMap<>();
    private final Map<String, T> modelsByKey = new ConcurrentHashMap<>();
    private final Map<Path, Object> modelLocks = new ConcurrentHashMap<>();

    Path resolve(Path modelPath) {
        Objects.requireNonNull(modelPath, "modelPath must not be null");
        return modelPath.toAbsolutePath().normalize();
    }

    T find(Path modelPath) {
        return modelsByPath.get(resolve(modelPath));
    }

    T findByKey(String key) {
        return modelsByKey.get(key);
    }

    boolean contains(Path modelPath) {
        return modelsByPath.containsKey(resolve(modelPath));
    }

    Object lockFor(Path modelPath) {
        Path resolved = resolve(modelPath);
        return modelLocks.computeIfAbsent(resolved, ignored -> new Object());
    }

    void register(Path modelPath, T model) {
        Objects.requireNonNull(model, "model must not be null");
        Path resolved = resolve(modelPath);
        modelsByPath.put(resolved, model);
        modelsByKey.put(model.key(), model);
    }

    T remove(Path modelPath) {
        T model = modelsByPath.remove(resolve(modelPath));
        if (model != null) {
            modelsByKey.remove(model.key());
        }
        return model;
    }

    Collection<T> snapshot() {
        return Collections.unmodifiableCollection(new ArrayList<>(modelsByPath.values()));
    }
}
