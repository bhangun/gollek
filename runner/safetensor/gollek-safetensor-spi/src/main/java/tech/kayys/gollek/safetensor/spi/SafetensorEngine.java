/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.spi;

import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.nio.file.Path;
import java.util.Map;

/**
 * SPI for SafeTensor inference engines.
 *
 * <p>Implementations manage the lifecycle of loaded SafeTensor models — loading
 * weights from disk, caching them in memory, and providing access by path or key.
 *
 * <p>Weight tensors are exposed as raw {@code Object} maps to avoid coupling the SPI
 * to a specific tensor backend (LibTorch, Accelerate, etc.). Implementations cast
 * to the concrete tensor type internally.
 *
 * @see SafetensorLoaderFacade
 */
public interface SafetensorEngine {

    /**
     * A fully-loaded SafeTensor model ready for inference.
     */
    interface LoadedModel {
        /** The filesystem path from which this model was loaded. */
        Path path();

        /** All weight tensors keyed by their parameter name. */
        Map<String, ?> weights();

        /** The tokenizer associated with this model. */
        Tokenizer tokenizer();

        /** Unique cache key for this model instance. */
        String key();

        /** {@code true} if the model weights have been quantized. */
        boolean isQuantized();

        /** Model architecture and hyperparameter configuration. */
        ModelConfig config();
    }

    /**
     * Loads a SafeTensor model from the given path and returns a cache key.
     *
     * @param modelPath path to a {@code .safetensors} file or model directory
     * @return a unique key that can be passed to {@link #getLoadedModel(String)}
     */
    String loadModel(Path modelPath);

    /**
     * Retrieves a previously loaded model by its filesystem path.
     *
     * @param modelPath the path used when calling {@link #loadModel}
     * @return the loaded model, or {@code null} if not found
     */
    LoadedModel getLoadedModel(Path modelPath);

    /**
     * Retrieves a previously loaded model by its cache key.
     *
     * @param key the key returned by {@link #loadModel}
     * @return the loaded model, or {@code null} if not found
     */
    LoadedModel getLoadedModel(String key);

    /**
     * Unloads a model and releases its associated resources.
     *
     * @param modelPath the path of the model to unload
     */
    void unloadModel(Path modelPath);
}
