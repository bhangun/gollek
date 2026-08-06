/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

class DirectGenerationOutputTest {

    @Test
    void convertsLoopResultToResponseShape() {
        DirectGenerationLoop.Result loop =
                new DirectGenerationLoop.Result("hello", List.of(3L, 5L), 2, 100L, 200L, 30L, 2);

        DirectGenerationOutput output = DirectGenerationOutput.fromLoop(loop);

        assertEquals("hello", output.text());
        assertArrayEquals(new long[] { 3L, 5L }, output.generatedTokenIds());
        assertEquals(2, output.generatedTokenCount());
        assertEquals(2, output.completionTokens());
    }

    @Test
    void recordsLoopTimingsWhenRequested() {
        DirectGenerationTimings timings = new DirectGenerationTimings();
        DirectGenerationLoop.Result loop =
                new DirectGenerationLoop.Result("", List.of(), 0, 50L, 80L, 20L, 1);

        DirectGenerationOutput.fromLoop(loop, timings);

        DirectInferenceResponses.BenchTimings bench = timings.benchTimings();
        assertEquals(50L, bench.firstTokenNanos());
        assertEquals(80L, bench.decodeNanos());
        assertEquals(20L, bench.samplingNanos());
        assertEquals(1, bench.decodeSteps());
    }

    @Test
    void handlesMissingLoopAsEmptyOutput() {
        DirectGenerationOutput output = DirectGenerationOutput.fromLoop(null);

        assertEquals("", output.text());
        assertArrayEquals(new long[0], output.generatedTokenIds());
        assertEquals(0, output.generatedTokenCount());
        assertEquals(0, output.completionTokens());
    }

    @Test
    void exposesSharedEmptyOutput() {
        assertSame(DirectGenerationOutput.empty(), DirectGenerationOutput.fromLoop(null));
    }

    @Test
    void fallsBackToSharedEmptyOutput() {
        DirectGenerationOutput output = new DirectGenerationOutput("x", new long[0], 0);

        assertSame(DirectGenerationOutput.empty(), DirectGenerationOutput.orEmpty(null));
        assertSame(output, DirectGenerationOutput.orEmpty(output));
    }
}
