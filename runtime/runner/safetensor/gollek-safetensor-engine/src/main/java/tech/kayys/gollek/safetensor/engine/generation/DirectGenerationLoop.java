/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import static tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler.markFirstToken;
import static tech.kayys.gollek.safetensor.engine.generation.GenerationTokenPolicy.recordSampleFrequency;

import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.StreamingDecoder;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Runs the token-by-token decode loop after prefill has selected the first
 * generated token.
 */
final class DirectGenerationLoop {
    private final Supplier<DirectGenerationStepSampler> stepSampler;

    DirectGenerationLoop(Supplier<DirectGenerationStepSampler> stepSampler) {
        this.stepSampler = stepSampler;
    }

    Result run(Request request) {
        DirectLoadedModel model = request.model();
        KVCacheManager.KVCacheSession session = request.session();
        DirectGenerationStepSampler.SamplingState sampling = request.sampling();
        Set<Integer> stops = request.stops();
        boolean checkStopTokens = stops != null && !stops.isEmpty();
        long requestStartNanos = request.requestStartNanos();
        InferenceProfile profile = request.profile();
        boolean collectTokenIds = request.collectTokenIds();
        boolean countCompletionTokens = request.countCompletionTokens();
        boolean normalizeNullDelta = request.normalizeNullDelta();
        BooleanSupplier cancelled = request.cancelled();
        Consumer<String> deltaConsumer = request.deltaConsumer();
        BiConsumer<Integer, Integer> nextTokenObserver = request.nextTokenObserver();
        StringBuilder out = new StringBuilder();
        List<Long> generatedIds = collectTokenIds ? new ArrayList<>() : Collections.emptyList();
        StreamingDecoder decoder = new StreamingDecoder(model.tokenizer(), DecodeOptions.defaultOptions());
        GenerationConfig cfg = request.cfg();
        List<String> stopStrings = cfg.stopStrings();
        boolean checkStopStrings = stopStrings != null && !stopStrings.isEmpty();
        int maxNewTokens = cfg.maxNewTokens();
        DirectGenerationStepSampler sampler = null;
        int next = request.firstToken();
        int completionTokens = 0;
        long firstTokenNanos = 0L;
        long decodeNanos = 0L;
        long samplingNanos = 0L;
        int decodeSteps = 0;

        System.err.println("DirectGenerationLoop: firstToken=" + next + ", stops=" + stops);
        for (int step = 0; step < maxNewTokens; step++) {
            if ((checkStopTokens && stops.contains(next)) || isCancelled(cancelled)) {
                break;
            }

            if (firstTokenNanos == 0L) {
                firstTokenNanos = System.nanoTime() - requestStartNanos;
                markFirstToken(profile, requestStartNanos);
            }
            if (collectTokenIds) {
                generatedIds.add((long) next);
            }
            if (countCompletionTokens) {
                completionTokens++;
            }

            String delta = decoder.decodeNext((long) next);
            if (delta == null && normalizeNullDelta) {
                delta = "";
            }
            out.append(delta);
            if (delta != null && !delta.isEmpty() && deltaConsumer != null) {
                deltaConsumer.accept(delta);
            }

            if (checkStopStrings && endsWithStopString(decoder.currentText(), stopStrings)) {
                break;
            }

            recordSampleFrequency(sampling.frequencies(), next);
            if (step + 1 >= maxNewTokens) {
                break;
            }
            if (sampler == null) {
                sampler = stepSampler.get();
            }
            DirectGenerationStepSampler.StepResult decode = sampler.decode(next,
                    request.decodeStartBase() + step, model, session, cfg, sampling);
            next = decode.token();
            decodeNanos += decode.forwardNanos();
            decodeSteps++;
            samplingNanos += decode.samplingNanos();
            if (profile != null) {
                profile.decodeNanos += decode.forwardNanos();
                profile.decodeSteps++;
                profile.samplingNanos += decode.samplingNanos();
            }
            if (nextTokenObserver != null) {
                nextTokenObserver.accept(next, step + 1);
            }
        }

        return new Result(out.toString(), generatedIds, completionTokens, firstTokenNanos, decodeNanos,
                samplingNanos, decodeSteps);
    }

    private static boolean endsWithStopString(String text, List<String> stopStrings) {
        if (stopStrings == null || stopStrings.isEmpty()) {
            return false;
        }
        for (String stop : stopStrings) {
            if (text.endsWith(stop)) {
                return true;
            }
        }
        return false;
    }

    private static boolean isCancelled(BooleanSupplier cancelled) {
        return cancelled != null && cancelled.getAsBoolean();
    }

    record Request(
            DirectLoadedModel model,
            GenerationConfig cfg,
            KVCacheManager.KVCacheSession session,
            DirectGenerationStepSampler.SamplingState sampling,
            Set<Integer> stops,
            int firstToken,
            int decodeStartBase,
            long requestStartNanos,
            InferenceProfile profile,
            boolean collectTokenIds,
            boolean countCompletionTokens,
            boolean normalizeNullDelta,
            BooleanSupplier cancelled,
            Consumer<String> deltaConsumer,
            BiConsumer<Integer, Integer> nextTokenObserver) {
    }

    record Result(String text, List<Long> generatedTokenIds, int completionTokens, long firstTokenNanos,
            long decodeNanos, long samplingNanos, int decodeSteps) {
    }
}
