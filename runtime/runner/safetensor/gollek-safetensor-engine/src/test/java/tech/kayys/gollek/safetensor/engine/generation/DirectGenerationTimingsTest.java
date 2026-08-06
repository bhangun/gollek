/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DirectGenerationTimingsTest {

    @Test
    void recordsSessionPrefillAndLoopTimings() {
        DirectGenerationTimings timings = new DirectGenerationTimings();
        InferenceProfile profile = new InferenceProfile("test", false);

        timings.recordSessionAllocate(100L, profile);
        timings.recordPrefill(new DirectGenerationStepSampler.StepResult(7, 200L, 30L), profile);
        timings.recordLoop(new DirectGenerationLoop.Result("ok", List.of(7L), 1, 500L, 300L, 40L, 2));

        DirectInferenceResponses.BenchTimings bench = timings.benchTimings();
        assertEquals(100L, bench.sessionAllocateNanos());
        assertEquals(200L, bench.prefillNanos());
        assertEquals(300L, bench.decodeNanos());
        assertEquals(70L, bench.samplingNanos());
        assertEquals(500L, bench.firstTokenNanos());
        assertEquals(2, bench.decodeSteps());

        assertEquals(100L, profile.sessionAllocateNanos);
        assertEquals(200L, profile.prefillNanos);
        assertEquals(30L, profile.samplingNanos);
    }

    @Test
    void clampsNegativeStandaloneTimings() {
        DirectGenerationTimings timings = new DirectGenerationTimings();

        timings.recordSessionAllocate(-1L, null);
        timings.recordPrefill(new DirectGenerationStepSampler.StepResult(7, -2L, -3L), null);

        DirectInferenceResponses.BenchTimings bench = timings.benchTimings();
        assertEquals(0L, bench.sessionAllocateNanos());
        assertEquals(0L, bench.prefillNanos());
        assertEquals(0L, bench.samplingNanos());
    }
}
