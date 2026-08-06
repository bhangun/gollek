/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import java.util.Map;

/**
 * Releases resources owned by a direct loaded model.
 */
final class DirectLoadedModelReleaser {
    private DirectLoadedModelReleaser() {
    }

    static void release(DirectInferenceEngine.LoadedModel model,
            ThrowingConsumer<Map<String, AccelTensor>> clearResolvedWeights) {
        if (model == null) {
            return;
        }
        release(model.weights(), model::closeWeightArena, clearResolvedWeights);
    }

    static <T extends AutoCloseable> void release(Map<String, T> weights, AutoCloseable arena,
            ThrowingConsumer<Map<String, T>> clearResolvedWeights) {
        if (weights != null && clearResolvedWeights != null) {
            try {
                clearResolvedWeights.accept(weights);
            } catch (Exception ignored) {
            }
        }
        if (weights != null) {
            weights.values().forEach(DirectLoadedModelReleaser::closeQuietly);
        }
        closeQuietly(arena);
    }

    private static void closeQuietly(AutoCloseable closeable) {
        if (closeable == null) {
            return;
        }
        try {
            closeable.close();
        } catch (Exception ignored) {
        }
    }

    @FunctionalInterface
    interface ThrowingConsumer<T> {
        void accept(T value) throws Exception;
    }
}
