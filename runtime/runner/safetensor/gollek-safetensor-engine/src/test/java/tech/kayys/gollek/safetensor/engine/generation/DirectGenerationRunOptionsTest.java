/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectGenerationRunOptionsTest {

    @Test
    void traceCollectsTokenIdsWithoutStreamingNormalization() {
        DirectGenerationRunOptions options = DirectGenerationRunOptions.trace();

        assertTrue(options.collectTokenIds());
        assertFalse(options.countCompletionTokens());
        assertFalse(options.normalizeNullDelta());
    }

    @Test
    void benchCountsCompletionTokensAndCanCarryObserver() {
        BiConsumer<Integer, Integer> observer = (token, step) -> { };

        DirectGenerationRunOptions options = DirectGenerationRunOptions.bench(observer);

        assertFalse(options.collectTokenIds());
        assertTrue(options.countCompletionTokens());
        assertSame(observer, options.nextTokenObserver());
    }

    @Test
    void streamTraceCarriesCancelAndDeltaHooks() {
        AtomicBoolean cancelled = new AtomicBoolean(true);
        AtomicReference<String> delta = new AtomicReference<>();

        DirectGenerationRunOptions options = DirectGenerationRunOptions.streamTrace(
                cancelled::get,
                delta::set);

        assertTrue(options.collectTokenIds());
        assertFalse(options.countCompletionTokens());
        assertTrue(options.normalizeNullDelta());
        assertTrue(options.cancelled().getAsBoolean());

        options.deltaConsumer().accept("hi");
        assertEquals("hi", delta.get());
    }
}
