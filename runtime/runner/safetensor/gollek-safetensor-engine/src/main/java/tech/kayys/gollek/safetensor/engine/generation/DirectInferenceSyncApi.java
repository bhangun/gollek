/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.inject.Instance;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.nio.file.Path;
import java.util.function.Supplier;

/**
 * Adapts direct generation flows into the public synchronous API.
 */
final class DirectInferenceSyncApi {
    private final Supplier<Instance<Object>> metalBackend;
    private final Logger log;
    private final DirectGenerationFlows.LoadedModelResolver modelResolver;
    private final Supplier<DirectGenerationFlows> generationFlows;

    DirectInferenceSyncApi(Supplier<Instance<Object>> metalBackend, Logger log,
            DirectGenerationFlows.LoadedModelResolver modelResolver,
            Supplier<DirectGenerationFlows> generationFlows) {
        this.metalBackend = metalBackend;
        this.log = log;
        this.modelResolver = modelResolver;
        this.generationFlows = generationFlows;
    }

    Uni<InferenceResponse> generate(String prompt, Path modelPath, GenerationConfig cfg) {
        return DirectSyncExecution.create(metalBackend.get(), log, "Generation failed", "Direct generation failed",
                request -> generationFlows.get().textResponse(request, prompt, modelPath, cfg));
    }

    long[] encodePrompt(String prompt, Path modelPath) {
        DirectInferenceEngine.LoadedModel model = modelResolver.require(modelPath);
        return DirectPromptTokens.encodeIds(model.tokenizer(), model.config(), model.runtimeTraits(), prompt, null);
    }

    Uni<DirectInferenceEngine.DirectGenerationTrace> generateWithTrace(long[] inputIds, Path modelPath,
            GenerationConfig cfg) {
        return DirectSyncExecution.create(metalBackend.get(), log, "Generation failed", "Direct generation failed",
                request -> generationFlows.get().pretokenizedTrace(request, inputIds, modelPath, cfg));
    }

    Uni<InferenceResponse> generate(long[] inputIds, Path modelPath, GenerationConfig cfg) {
        return generateWithTrace(inputIds, modelPath, cfg)
                .map(DirectInferenceEngine.DirectGenerationTrace::response);
    }

    Uni<DirectInferenceEngine.DirectConversationTrace> generateWithConversationTrace(long[] inputIds, Path modelPath,
            GenerationConfig cfg) {
        return DirectSyncExecution.create(metalBackend.get(), log, "Conversation generation failed",
                "Direct conversation generation failed",
                request -> generationFlows.get().conversationTrace(request, inputIds, modelPath, cfg));
    }

    Uni<DirectInferenceEngine.DirectConversationTrace> generateContinuationWithConversationTrace(
            long[] fullInputIds,
            int cachedPrefixTokens,
            KVCacheManager.KVCacheSession session,
            Path modelPath,
            GenerationConfig cfg,
            Integer replayTokenId) {
        return DirectSyncExecution.create(metalBackend.get(), log, "Conversation continuation failed",
                "Direct conversation continuation failed",
                request -> generationFlows.get().continuationTrace(
                        request, fullInputIds, cachedPrefixTokens, session, modelPath, cfg, replayTokenId));
    }
}
