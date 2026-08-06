/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.nio.file.Path;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Coordinates streaming direct-generation flows after the worker envelope has
 * been created.
 */
final class DirectStreamingGenerationFlows {
    private final DirectGenerationFlows.LoadedModelResolver modelResolver;
    private final Supplier<DirectGenerationSessionAllocator> sessionAllocator;
    private final Supplier<DirectGenerationExecutor> generationExecutor;

    DirectStreamingGenerationFlows(DirectGenerationFlows.LoadedModelResolver modelResolver,
            Supplier<DirectGenerationSessionAllocator> sessionAllocator,
            Supplier<DirectGenerationExecutor> generationExecutor) {
        this.modelResolver = modelResolver;
        this.sessionAllocator = sessionAllocator;
        this.generationExecutor = generationExecutor;
    }

    void conversationTrace(DirectGenerationRequestContext request,
            long[] inputIds,
            Path modelPath,
            GenerationConfig cfg,
            Consumer<DirectInferenceEngine.DirectConversationTrace> onComplete,
            StreamSink sink) {
        Objects.requireNonNull(sink, "sink");
        DirectInferenceEngine.LoadedModel model = modelResolver.require(modelPath);
        DirectPromptPreparation preparedPrompt = DirectPromptPreparation.pretokenized(
                model, inputIds, DirectPromptPreparation.DebugOptions.none());
        ModelConfig config = preparedPrompt.config();

        try (DirectConversationSessionOwner sessionOwner = DirectConversationSessionOwner.of(
                sessionAllocator.get().createAllocated(config, cfg, request.timings(), request.profile()))) {
            KVCacheManager.KVCacheSession session = sessionOwner.session();

            Set<Integer> stops = requestStopTokenIds(model, cfg);

            Consumer<String> deltaConsumer = DirectStreamResponses.deltaConsumer(
                    sink::emit, request, modelPath, preparedPrompt,
                    DirectGenerationMetadata.retainedPretokenizedConversation());
            DirectGenerationLoop.Result loop = generationExecutor.get().runPrefill(
                    new DirectGenerationExecutor.PrefillRequest(
                            model, cfg, session, preparedPrompt.ids(), stops,
                            DirectGenerationStepSampler.SamplingMode.RAW_PRETOKENIZED,
                            preparedPrompt.length(), request.startedNanos(), request.profile(),
                            request.timings(),
                            DirectGenerationRunOptions.streamTrace(
                                    sink::isCancelled, deltaConsumer)))
                    .loop();
            DirectGenerationOutput output = DirectGenerationOutput.fromLoop(loop);

            DirectInferenceEngine.DirectConversationTrace trace = DirectGenerationTraces.streamConversationTrace(
                    request, output, modelPath, preparedPrompt, session,
                    DirectGenerationMetadata.retainedPretokenizedConversation());
            sessionOwner.transferTo(onComplete, trace);
            sink.emit(trace.response());
        }
    }

    void continuationTrace(DirectGenerationRequestContext request,
            long[] fullInputIds,
            int cachedPrefixTokens,
            KVCacheManager.KVCacheSession session,
            Path modelPath,
            GenerationConfig cfg,
            Consumer<DirectInferenceEngine.DirectConversationTrace> onComplete,
            Integer replayTokenId,
            StreamSink sink) {
        Objects.requireNonNull(sink, "sink");
        if (session == null) {
            throw new IllegalArgumentException("Conversation continuation requires an active KV cache session");
        }
        DirectConversationContinuationPlan continuation = DirectConversationContinuationPlan.resolve(
                fullInputIds, cachedPrefixTokens, session.currentPos(), replayTokenId);
        int inputLen = continuation.fullInputIds().length;

        DirectInferenceEngine.LoadedModel model = modelResolver.require(modelPath);
        Set<Integer> stops = requestStopTokenIds(model, cfg);
        Consumer<String> deltaConsumer = DirectStreamResponses.deltaConsumer(
                sink::emit, request, modelPath, inputLen, continuation.retainedKvMetadata());
        DirectGenerationLoop.Result loop = generationExecutor.get().runContinuation(
                new DirectGenerationExecutor.ContinuationRequest(
                        model, cfg, session, continuation, stops,
                        request.startedNanos(), request.profile(), request.timings(),
                        DirectGenerationRunOptions.streamTrace(
                                sink::isCancelled, deltaConsumer)))
                .loop();
        DirectGenerationOutput output = DirectGenerationOutput.fromLoop(loop);

        DirectInferenceEngine.DirectConversationTrace trace = DirectGenerationTraces.streamContinuationTrace(
                request, output, modelPath, continuation, session);
        if (onComplete != null) {
            onComplete.accept(trace);
        }
        sink.emit(trace.response());
    }

    void textResponse(DirectGenerationRequestContext request, String prompt,
            Path modelPath, GenerationConfig cfg, StreamSink sink) {
        Objects.requireNonNull(sink, "sink");
        DirectGenerationDebug debug = DirectGenerationDebug.stream();
        DirectInferenceEngine.LoadedModel model =
                modelResolver.require(modelPath, debug.enabled(), debug.prefix());
        DirectPromptPreparation preparedPrompt = DirectPromptPreparation.text(
                model, prompt, request.profile(), debug.textPrompt(true));
        ModelConfig config = preparedPrompt.config();
        long[] inputIds = preparedPrompt.ids();
        int inputLen = preparedPrompt.length();

        Set<Integer> stops = requestStopTokenIds(model, cfg);

        DirectGenerationOutput output;
        debug.step(7, "allocate session");
        try (KVCacheManager.KVCacheSession session = sessionAllocator.get()
                .createAllocated(config, cfg, request.timings(), request.profile())) {
            debug.step(8, "prefill");
            Consumer<String> deltaConsumer = DirectStreamResponses.deltaConsumer(
                    sink::emit, request, modelPath, inputLen, null);
            DirectGenerationExecutor.Result run = generationExecutor.get().runPrefill(
                    new DirectGenerationExecutor.PrefillRequest(
                            model, cfg, session, inputIds, stops,
                            DirectGenerationStepSampler.SamplingMode.TOKENIZER_AWARE,
                            inputIds.length, request.startedNanos(), request.profile(), request.timings(),
                            DirectGenerationRunOptions.streamBench(
                                    sink::isCancelled, deltaConsumer)));
            debug.step(9, "prefill done");

            output = DirectGenerationOutput.fromLoop(run.loop(), request.timings());
        }

        sink.emit(DirectInferenceResponses.finalBenchResponse(
                request, "", output, modelPath, inputLen, null));
    }

    void pretokenizedResponse(DirectGenerationRequestContext request, long[] inputIds,
            Path modelPath, GenerationConfig cfg, StreamSink sink) {
        Objects.requireNonNull(sink, "sink");
        DirectInferenceEngine.LoadedModel model = modelResolver.require(modelPath);
        DirectPromptPreparation preparedPrompt = DirectPromptPreparation.pretokenized(
                model, inputIds, DirectPromptPreparation.DebugOptions.none());
        ModelConfig config = preparedPrompt.config();
        int inputLen = preparedPrompt.length();

        DirectGenerationOutput output;
        try (KVCacheManager.KVCacheSession session = sessionAllocator.get()
                .createAllocated(config, cfg, request.timings(), request.profile())) {
            Set<Integer> stops = requestStopTokenIds(model, cfg);

            Consumer<String> deltaConsumer = DirectStreamResponses.deltaConsumer(
                    sink::emit, request, modelPath, preparedPrompt,
                    DirectGenerationMetadata.pretokenizedPrompt());
            DirectGenerationLoop.Result loop = generationExecutor.get().runPrefill(
                    new DirectGenerationExecutor.PrefillRequest(
                            model, cfg, session, preparedPrompt.ids(), stops,
                            DirectGenerationStepSampler.SamplingMode.TOKENIZER_AWARE,
                            preparedPrompt.length(), request.startedNanos(), request.profile(),
                            request.timings(),
                            DirectGenerationRunOptions.streamBench(
                                    sink::isCancelled, deltaConsumer)))
                    .loop();
            output = DirectGenerationOutput.fromLoop(loop, request.timings());
        }

        sink.emit(DirectInferenceResponses.finalBenchResponse(
                request, "", output, modelPath, inputLen,
                DirectGenerationMetadata.pretokenizedPrompt()));
    }

    private static Set<Integer> requestStopTokenIds(DirectInferenceEngine.LoadedModel model, GenerationConfig cfg) {
        return GenerationTokenPolicy.mergeStopTokenIds(model.baseStopTokenIds(), cfg);
    }

    interface StreamSink {
        void emit(InferenceResponse response);

        boolean isCancelled();
    }
}
