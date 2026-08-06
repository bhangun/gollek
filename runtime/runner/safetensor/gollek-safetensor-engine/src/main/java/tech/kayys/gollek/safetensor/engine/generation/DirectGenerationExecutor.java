/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;

import java.util.Set;
import java.util.function.Supplier;

/**
 * Coordinates sampling-state creation, prefill, and decode-loop execution.
 */
final class DirectGenerationExecutor {
    private final StepSampler stepSampler;
    private final LoopRunner loopRunner;

    DirectGenerationExecutor(Supplier<DirectGenerationStepSampler> stepSampler,
            Supplier<DirectGenerationLoop> generationLoop) {
        this(new StepSampler() {
            @Override
            public DirectGenerationStepSampler.SamplingState createState(
                    DirectLoadedModel model,
                    GenerationConfig cfg,
                    long[] promptTokenIds,
                    Set<Integer> stops,
                    DirectGenerationStepSampler.SamplingMode mode) {
                return stepSampler.get().createState(model, cfg, promptTokenIds, stops, mode);
            }

            @Override
            public DirectGenerationStepSampler.StepResult prefill(
                    long[] inputIds,
                    DirectLoadedModel model,
                    KVCacheManager.KVCacheSession session,
                    GenerationConfig cfg,
                    DirectGenerationStepSampler.SamplingState sampling) {
                return stepSampler.get().prefill(inputIds, model, session, cfg, sampling);
            }
        }, request -> generationLoop.get().run(request));
    }

    DirectGenerationExecutor(StepSampler stepSampler, LoopRunner loopRunner) {
        this.stepSampler = stepSampler;
        this.loopRunner = loopRunner;
    }

    Result runPrefill(PrefillRequest request) {
        DirectGenerationStepSampler.SamplingState sampling = stepSampler.createState(
                request.model(), request.cfg(), request.inputIds(), request.stops(), request.samplingMode());
        DirectGenerationStepSampler.StepResult prefill = stepSampler.prefill(
                request.inputIds(), request.model(), request.session(), request.cfg(), sampling);
        recordPrefill(request.timings(), prefill, request.profile());

        DirectGenerationLoop.Result loop = loopRunner.run(new DirectGenerationLoop.Request(
                request.model(),
                request.cfg(),
                request.session(),
                sampling,
                request.stops(),
                prefill.token(),
                request.decodeStartBase(),
                request.requestStartNanos(),
                request.profile(),
                request.options().collectTokenIds(),
                request.options().countCompletionTokens(),
                request.options().normalizeNullDelta(),
                request.options().cancelled(),
                request.options().deltaConsumer(),
                request.options().nextTokenObserver()));
        return new Result(prefill, loop);
    }

    Result runContinuation(ContinuationRequest request) {
        DirectGenerationStepSampler.SamplingState sampling = stepSampler.createState(
                request.model(),
                request.cfg(),
                request.continuation().fullInputIds(),
                request.stops(),
                DirectGenerationStepSampler.SamplingMode.RAW_PRETOKENIZED);
        DirectGenerationStepSampler.StepResult prefill = null;
        int next;
        if (request.continuation().exactReplay()) {
            next = request.continuation().nextReplayTokenId();
        } else {
            prefill = stepSampler.prefill(
                    request.continuation().deltaInputIds(),
                    request.model(),
                    request.session(),
                    request.cfg(),
                    sampling);
            next = prefill.token();
            recordPrefill(request.timings(), prefill, request.profile());
        }

        DirectGenerationLoop.Result loop = loopRunner.run(new DirectGenerationLoop.Request(
                request.model(),
                request.cfg(),
                request.session(),
                sampling,
                request.stops(),
                next,
                request.continuation().fullInputIds().length,
                request.requestStartNanos(),
                request.profile(),
                request.options().collectTokenIds(),
                request.options().countCompletionTokens(),
                request.options().normalizeNullDelta(),
                request.options().cancelled(),
                request.options().deltaConsumer(),
                request.options().nextTokenObserver()));
        return new Result(prefill, loop);
    }

    private static void recordPrefill(DirectGenerationTimings timings,
            DirectGenerationStepSampler.StepResult prefill,
            InferenceProfile profile) {
        if (timings != null) {
            timings.recordPrefill(prefill, profile);
        }
    }

    interface StepSampler {
        DirectGenerationStepSampler.SamplingState createState(
                DirectLoadedModel model,
                GenerationConfig cfg,
                long[] promptTokenIds,
                Set<Integer> stops,
                DirectGenerationStepSampler.SamplingMode mode);

        DirectGenerationStepSampler.StepResult prefill(
                long[] inputIds,
                DirectLoadedModel model,
                KVCacheManager.KVCacheSession session,
                GenerationConfig cfg,
                DirectGenerationStepSampler.SamplingState sampling);
    }

    interface LoopRunner {
        DirectGenerationLoop.Result run(DirectGenerationLoop.Request request);
    }

    record PrefillRequest(
            DirectLoadedModel model,
            GenerationConfig cfg,
            KVCacheManager.KVCacheSession session,
            long[] inputIds,
            Set<Integer> stops,
            DirectGenerationStepSampler.SamplingMode samplingMode,
            int decodeStartBase,
            long requestStartNanos,
            InferenceProfile profile,
            DirectGenerationTimings timings,
            DirectGenerationRunOptions options) {
        PrefillRequest {
            options = DirectGenerationRunOptions.orDefault(options);
        }
    }

    record ContinuationRequest(
            DirectLoadedModel model,
            GenerationConfig cfg,
            KVCacheManager.KVCacheSession session,
            DirectConversationContinuationPlan continuation,
            Set<Integer> stops,
            long requestStartNanos,
            InferenceProfile profile,
            DirectGenerationTimings timings,
            DirectGenerationRunOptions options) {
        ContinuationRequest {
            options = DirectGenerationRunOptions.orDefault(options);
        }
    }

    record Result(
            DirectGenerationStepSampler.StepResult prefill,
            DirectGenerationLoop.Result loop) {
    }
}
