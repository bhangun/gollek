/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import java.util.function.BiConsumer;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

/**
 * Runtime decode-loop options shared by prefill and continuation execution.
 */
record DirectGenerationRunOptions(
        boolean collectTokenIds,
        boolean countCompletionTokens,
        boolean normalizeNullDelta,
        BooleanSupplier cancelled,
        Consumer<String> deltaConsumer,
        BiConsumer<Integer, Integer> nextTokenObserver) {

    private static final DirectGenerationRunOptions TRACE =
            new DirectGenerationRunOptions(true, false, false, null, null, null);

    private static final DirectGenerationRunOptions BENCH =
            new DirectGenerationRunOptions(false, true, false, null, null, null);

    static DirectGenerationRunOptions trace() {
        return TRACE;
    }

    static DirectGenerationRunOptions bench() {
        return BENCH;
    }

    static DirectGenerationRunOptions bench(BiConsumer<Integer, Integer> nextTokenObserver) {
        if (nextTokenObserver == null) {
            return BENCH;
        }
        return new DirectGenerationRunOptions(false, true, false, null, null, nextTokenObserver);
    }

    static DirectGenerationRunOptions streamTrace(BooleanSupplier cancelled, Consumer<String> deltaConsumer) {
        return new DirectGenerationRunOptions(true, false, true, cancelled, deltaConsumer, null);
    }

    static DirectGenerationRunOptions streamBench(BooleanSupplier cancelled, Consumer<String> deltaConsumer) {
        return new DirectGenerationRunOptions(false, true, true, cancelled, deltaConsumer, null);
    }

    static DirectGenerationRunOptions orDefault(DirectGenerationRunOptions options) {
        return options == null ? BENCH : options;
    }
}
