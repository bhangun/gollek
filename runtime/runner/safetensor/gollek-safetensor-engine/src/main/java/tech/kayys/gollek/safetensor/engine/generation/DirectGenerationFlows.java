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
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.nio.file.Path;
import java.util.Set;
import java.util.function.Supplier;

/**
 * Coordinates synchronous direct-generation flows after the request envelope
 * has been created.
 */
final class DirectGenerationFlows {
    private final LoadedModelResolver modelResolver;
    private final Supplier<DirectGenerationSessionAllocator> sessionAllocator;
    private final Supplier<DirectGenerationExecutor> generationExecutor;

    DirectGenerationFlows(LoadedModelResolver modelResolver,
            Supplier<DirectGenerationSessionAllocator> sessionAllocator,
            Supplier<DirectGenerationExecutor> generationExecutor) {
        this.modelResolver = modelResolver;
        this.sessionAllocator = sessionAllocator;
        this.generationExecutor = generationExecutor;
    }

    InferenceResponse textResponse(DirectGenerationRequestContext request, String prompt,
            Path modelPath, GenerationConfig cfg) {
        DirectGenerationDebug debug = DirectGenerationDebug.text();
        DirectInferenceEngine.LoadedModel model =
                modelResolver.require(modelPath, debug.enabled(), debug.prefix());
        DirectPromptPreparation preparedPrompt = DirectPromptPreparation.text(
                model, prompt, request.profile(), debug.textPrompt(false));
        Tokenizer tokenizer = preparedPrompt.tokenizer();
        ModelConfig config = preparedPrompt.config();
        long[] inputIds = preparedPrompt.ids();
        int inputLen = preparedPrompt.length();

        Set<Integer> stops = requestStopTokenIds(model, cfg);

        debug.step(7, "allocate session");
        DirectGenerationOutput output;
        try (KVCacheManager.KVCacheSession session = sessionAllocator.get()
                .createAllocated(config, cfg, request.timings(), request.profile())) {
            debug.step(8, "prefill");
            DirectGenerationExecutor.Result run = generationExecutor.get().runPrefill(
                    new DirectGenerationExecutor.PrefillRequest(
                            model, cfg, session, inputIds, stops,
                            DirectGenerationStepSampler.SamplingMode.TOKENIZER_AWARE,
                            inputIds.length, request.startedNanos(), request.profile(), request.timings(),
                            DirectGenerationRunOptions.bench(debug.chosenTokenObserver(tokenizer))));
            debug.step(9, "prefill done");
            debug.chosenToken(tokenizer, run.prefill().token(), 0);

            output = DirectGenerationOutput.fromLoop(run.loop(), request.timings());
        }

        return DirectInferenceResponses.finalBenchResponse(request, output, modelPath, inputLen, null);
    }

    DirectInferenceEngine.DirectGenerationTrace pretokenizedTrace(
            DirectGenerationRequestContext request, long[] inputIds, Path modelPath, GenerationConfig cfg) {
        DirectGenerationDebug debug = DirectGenerationDebug.text();
        DirectInferenceEngine.LoadedModel model = modelResolver.require(modelPath);
        DirectPromptPreparation preparedPrompt = DirectPromptPreparation.pretokenized(
                model, inputIds, debug.pretokenized("[DEBUG] pretokenized"));
        ModelConfig config = preparedPrompt.config();

        try (KVCacheManager.KVCacheSession session = sessionAllocator.get()
                .createAllocated(config, cfg, request.timings(), request.profile())) {
            Set<Integer> stops = requestStopTokenIds(model, cfg);
            DirectGenerationLoop.Result loop = generationExecutor.get().runPrefill(
                    new DirectGenerationExecutor.PrefillRequest(
                            model, cfg, session, preparedPrompt.ids(), stops,
                            DirectGenerationStepSampler.SamplingMode.RAW_PRETOKENIZED,
                            preparedPrompt.length(), request.startedNanos(), request.profile(),
                            request.timings(),
                            DirectGenerationRunOptions.trace()))
                    .loop();
            DirectGenerationOutput output = DirectGenerationOutput.fromLoop(loop);
            return DirectGenerationTraces.generationTrace(
                    request, output, modelPath, preparedPrompt,
                    DirectGenerationMetadata.pretokenizedPrompt());
        }
    }

    DirectInferenceEngine.DirectConversationTrace conversationTrace(
            DirectGenerationRequestContext request, long[] inputIds, Path modelPath, GenerationConfig cfg) {
        DirectGenerationDebug debug = DirectGenerationDebug.text();
        DirectInferenceEngine.LoadedModel model = modelResolver.require(modelPath);
        DirectPromptPreparation preparedPrompt = DirectPromptPreparation.pretokenized(
                model, inputIds, debug.pretokenized("[DEBUG] conversational pretokenized"));
        ModelConfig config = preparedPrompt.config();

        try (DirectConversationSessionOwner sessionOwner = DirectConversationSessionOwner.of(
                sessionAllocator.get().createAllocated(config, cfg, request.timings(), request.profile()))) {
            KVCacheManager.KVCacheSession session = sessionOwner.session();

            Set<Integer> stops = requestStopTokenIds(model, cfg);
            DirectGenerationLoop.Result loop = generationExecutor.get().runPrefill(
                    new DirectGenerationExecutor.PrefillRequest(
                            model, cfg, session, preparedPrompt.ids(), stops,
                            DirectGenerationStepSampler.SamplingMode.RAW_PRETOKENIZED,
                            preparedPrompt.length(), request.startedNanos(), request.profile(),
                            request.timings(),
                            DirectGenerationRunOptions.trace()))
                    .loop();
            DirectGenerationOutput output = DirectGenerationOutput.fromLoop(loop);

            DirectInferenceEngine.DirectConversationTrace trace = DirectGenerationTraces.conversationTrace(
                    request, output, modelPath, preparedPrompt, session,
                    DirectGenerationMetadata.retainedPretokenizedConversation());
            return sessionOwner.transfer(trace);
        }
    }

    DirectInferenceEngine.DirectConversationTrace continuationTrace(
            DirectGenerationRequestContext request,
            long[] fullInputIds,
            int cachedPrefixTokens,
            KVCacheManager.KVCacheSession session,
            Path modelPath,
            GenerationConfig cfg,
            Integer replayTokenId) {
        if (session == null) {
            throw new IllegalArgumentException("Conversation continuation requires an active KV cache session");
        }
        DirectConversationContinuationPlan continuation = DirectConversationContinuationPlan.resolve(
                fullInputIds, cachedPrefixTokens, session.currentPos(), replayTokenId);

        DirectInferenceEngine.LoadedModel model = modelResolver.require(modelPath);

        Set<Integer> stops = requestStopTokenIds(model, cfg);
        DirectGenerationLoop.Result loop = generationExecutor.get().runContinuation(
                new DirectGenerationExecutor.ContinuationRequest(
                        model, cfg, session, continuation, stops,
                        request.startedNanos(), request.profile(), request.timings(),
                        DirectGenerationRunOptions.trace()))
                .loop();
        DirectGenerationOutput output = DirectGenerationOutput.fromLoop(loop);

        return DirectGenerationTraces.continuationTrace(request, output, modelPath, continuation, session);
    }

    private static Set<Integer> requestStopTokenIds(DirectInferenceEngine.LoadedModel model, GenerationConfig cfg) {
        return GenerationTokenPolicy.mergeStopTokenIds(model.baseStopTokenIds(), cfg);
    }

    @FunctionalInterface
    interface LoadedModelResolver {
        DirectInferenceEngine.LoadedModel require(Path modelPath, boolean verbose, String debugPrefix);

        default DirectInferenceEngine.LoadedModel require(Path modelPath) {
            return require(modelPath, false, null);
        }
    }
}
