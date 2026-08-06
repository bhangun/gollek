/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.subscription.MultiEmitter;
import jakarta.enterprise.inject.Instance;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.nio.file.Path;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.concurrent.ThreadFactory;

/**
 * Adapts direct generation flows into the public streaming API.
 */
final class DirectInferenceStreamingApi {
    private static final ThreadFactory STREAM_EXECUTOR_THREAD_FACTORY = runnable -> {
        Thread thread = new Thread(runnable, "gollek-direct-stream");
        thread.setDaemon(true);
        return thread;
    };

    private final Supplier<Instance<Object>> metalBackend;
    private final Logger log;
    private final Supplier<DirectStreamingGenerationFlows> streamingGenerationFlows;

    DirectInferenceStreamingApi(Supplier<Instance<Object>> metalBackend, Logger log,
            Supplier<DirectStreamingGenerationFlows> streamingGenerationFlows) {
        this.metalBackend = metalBackend;
        this.log = log;
        this.streamingGenerationFlows = streamingGenerationFlows;
    }

    Multi<InferenceResponse> generateWithConversationTrace(long[] inputIds, Path modelPath, GenerationConfig cfg,
            Consumer<DirectInferenceEngine.DirectConversationTrace> onComplete) {
        return DirectStreamExecution.create(STREAM_EXECUTOR_THREAD_FACTORY, log,
                "Streaming conversation generation failed", DirectInferenceProfiler::clearProfile,
                emitter -> {
                    DirectGenerationRequestContext request = requestContext();
                    streamingGenerationFlows.get().conversationTrace(
                            request, inputIds, modelPath, cfg, onComplete, streamSink(emitter));
                });
    }

    Multi<InferenceResponse> generateContinuationWithConversationTrace(
            long[] fullInputIds,
            int cachedPrefixTokens,
            KVCacheManager.KVCacheSession session,
            Path modelPath,
            GenerationConfig cfg,
            Consumer<DirectInferenceEngine.DirectConversationTrace> onComplete,
            Integer replayTokenId) {
        return DirectStreamExecution.create(STREAM_EXECUTOR_THREAD_FACTORY, log,
                "Streaming conversation continuation failed", DirectInferenceProfiler::clearProfile,
                emitter -> {
                    DirectGenerationRequestContext request = requestContext();
                    streamingGenerationFlows.get().continuationTrace(
                            request, fullInputIds, cachedPrefixTokens, session, modelPath, cfg, onComplete,
                            replayTokenId, streamSink(emitter));
                });
    }

    Multi<InferenceResponse> generate(String prompt, Path modelPath, GenerationConfig cfg) {
        return DirectStreamExecution.create(STREAM_EXECUTOR_THREAD_FACTORY, log,
                "Generation failed", DirectInferenceProfiler::clearProfile, emitter -> {
                    DirectGenerationRequestContext request = requestContext();
                    streamingGenerationFlows.get().textResponse(
                            request, prompt, modelPath, cfg, streamSink(emitter));
                });
    }

    Multi<InferenceResponse> generate(long[] inputIds, Path modelPath, GenerationConfig cfg) {
        return DirectStreamExecution.create(STREAM_EXECUTOR_THREAD_FACTORY, log,
                "Generation failed", DirectInferenceProfiler::clearProfile, emitter -> {
                    DirectGenerationRequestContext request = requestContext();
                    streamingGenerationFlows.get().pretokenizedResponse(
                            request, inputIds, modelPath, cfg, streamSink(emitter));
                });
    }

    private DirectGenerationRequestContext requestContext() {
        return DirectGenerationRequestContext.stream(metalBackend.get());
    }

    private static DirectStreamingGenerationFlows.StreamSink streamSink(
            MultiEmitter<? super InferenceResponse> emitter) {
        return new DirectStreamingGenerationFlows.StreamSink() {
            @Override
            public void emit(InferenceResponse response) {
                emitter.emit(response);
            }

            @Override
            public boolean isCancelled() {
                return emitter.isCancelled();
            }
        };
    }
}
