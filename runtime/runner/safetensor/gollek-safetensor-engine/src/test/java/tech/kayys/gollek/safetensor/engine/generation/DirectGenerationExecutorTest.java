/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;

import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

class DirectGenerationExecutorTest {

    @Test
    void prefillCreatesSamplingStateRecordsTimingAndRunsLoop() {
        GenerationConfig cfg = GenerationConfig.defaults();
        Set<Integer> stops = Set.of(2);
        long[] inputIds = { 10, 11, 12 };
        InferenceProfile profile = new InferenceProfile("test", false);
        DirectGenerationTimings timings = new DirectGenerationTimings();
        DirectGenerationStepSampler.SamplingState sampling = sampling(stops);
        AtomicBoolean createdState = new AtomicBoolean(false);

        DirectGenerationExecutor executor = new DirectGenerationExecutor(new DirectGenerationExecutor.StepSampler() {
            @Override
            public DirectGenerationStepSampler.SamplingState createState(
                    DirectLoadedModel model,
                    GenerationConfig requestedCfg,
                    long[] promptTokenIds,
                    Set<Integer> requestedStops,
                    DirectGenerationStepSampler.SamplingMode mode) {
                assertSame(cfg, requestedCfg);
                assertArrayEquals(inputIds, promptTokenIds);
                assertSame(stops, requestedStops);
                assertEquals(DirectGenerationStepSampler.SamplingMode.RAW_PRETOKENIZED, mode);
                createdState.set(true);
                return sampling;
            }

            @Override
            public DirectGenerationStepSampler.StepResult prefill(
                    long[] requestedInputIds,
                    DirectLoadedModel model,
                    tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager.KVCacheSession session,
                    GenerationConfig requestedCfg,
                    DirectGenerationStepSampler.SamplingState requestedSampling) {
                assertArrayEquals(inputIds, requestedInputIds);
                assertSame(sampling, requestedSampling);
                return new DirectGenerationStepSampler.StepResult(42, 100L, 7L);
            }
        }, request -> {
            assertEquals(42, request.firstToken());
            assertEquals(inputIds.length, request.decodeStartBase());
            assertSame(profile, request.profile());
            assertTrue(request.collectTokenIds());
            assertFalse(request.countCompletionTokens());
            return new DirectGenerationLoop.Result("ok", List.of(42L), 0, 1L, 2L, 3L, 1);
        });

        DirectGenerationExecutor.Result result = executor.runPrefill(
                new DirectGenerationExecutor.PrefillRequest(
                        null, cfg, null, inputIds, stops,
                        DirectGenerationStepSampler.SamplingMode.RAW_PRETOKENIZED,
                        inputIds.length, 123L, profile, timings,
                        DirectGenerationRunOptions.trace()));

        assertTrue(createdState.get());
        assertEquals(42, result.prefill().token());
        assertEquals("ok", result.loop().text());
        assertEquals(100L, timings.benchTimings().prefillNanos());
        assertEquals(7L, profile.samplingNanos);
    }

    @Test
    void exactReplayContinuationSkipsPrefillAndUsesReplayToken() {
        GenerationConfig cfg = GenerationConfig.defaults();
        Set<Integer> stops = Set.of();
        DirectConversationContinuationPlan continuation =
                DirectConversationContinuationPlan.resolve(new long[] { 1, 2 }, 2, 2, 99);
        DirectGenerationStepSampler.SamplingState sampling = sampling(stops);
        AtomicBoolean prefillCalled = new AtomicBoolean(false);

        DirectGenerationExecutor executor = new DirectGenerationExecutor(new DirectGenerationExecutor.StepSampler() {
            @Override
            public DirectGenerationStepSampler.SamplingState createState(
                    DirectLoadedModel model,
                    GenerationConfig requestedCfg,
                    long[] promptTokenIds,
                    Set<Integer> requestedStops,
                    DirectGenerationStepSampler.SamplingMode mode) {
                assertArrayEquals(new long[] { 1, 2 }, promptTokenIds);
                assertEquals(DirectGenerationStepSampler.SamplingMode.RAW_PRETOKENIZED, mode);
                return sampling;
            }

            @Override
            public DirectGenerationStepSampler.StepResult prefill(
                    long[] requestedInputIds,
                    DirectLoadedModel model,
                    tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager.KVCacheSession session,
                    GenerationConfig requestedCfg,
                    DirectGenerationStepSampler.SamplingState requestedSampling) {
                prefillCalled.set(true);
                fail("exact replay should not run prefill");
                return null;
            }
        }, request -> {
            assertEquals(99, request.firstToken());
            assertEquals(2, request.decodeStartBase());
            return new DirectGenerationLoop.Result("", List.of(99L), 0, 0L, 0L, 0L, 0);
        });

        DirectGenerationExecutor.Result result = executor.runContinuation(
                new DirectGenerationExecutor.ContinuationRequest(
                        null, cfg, null, continuation, stops,
                        123L, null, new DirectGenerationTimings(),
                        DirectGenerationRunOptions.streamTrace(null, null)));

        assertFalse(prefillCalled.get());
        assertEquals(99L, result.loop().generatedTokenIds().get(0));
    }

    private static DirectGenerationStepSampler.SamplingState sampling(Set<Integer> stops) {
        return new DirectGenerationStepSampler.SamplingState(
                DirectGenerationStepSampler.SamplingMode.RAW_PRETOKENIZED,
                false, null, null, null, null, stops, null, null);
    }
}
