/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import java.util.function.LongConsumer;
import java.util.function.Supplier;

/**
 * Common model acquisition flow for direct generation entry points.
 */
final class DirectLoadedModelAcquirer {
    private DirectLoadedModelAcquirer() {
    }

    static <T> T require(Supplier<T> lookup, Runnable load, LongConsumer recordLoadNanos,
            boolean verbose, String debugPrefix) {
        debugStep(verbose, debugPrefix, 1, "getLoadedModel");
        T model = lookup.get();
        if (model == null) {
            debugStep(verbose, debugPrefix, 2, "loadModel");
            long tLoad0 = System.nanoTime();
            load.run();
            if (recordLoadNanos != null) {
                recordLoadNanos.accept(System.nanoTime() - tLoad0);
            }
            model = lookup.get();
        }
        if (model == null) {
            throw new RuntimeException("Model failed to load");
        }
        return model;
    }

    private static void debugStep(boolean verbose, String prefix, int step, String message) {
        if (!verbose) {
            return;
        }
        String label = prefix == null || prefix.isBlank() ? "[DEBUG]" : prefix;
        System.out.println(label + " " + step + ": " + message);
        System.out.flush();
    }
}
