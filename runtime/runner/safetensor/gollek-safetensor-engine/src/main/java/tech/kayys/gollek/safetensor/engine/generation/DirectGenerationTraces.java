/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.nio.file.Path;
import java.util.Map;

/**
 * Builds trace payloads from completed generation output.
 */
final class DirectGenerationTraces {

    private DirectGenerationTraces() {
    }

    static DirectInferenceEngine.DirectGenerationTrace generationTrace(
            DirectGenerationRequestContext request,
            DirectGenerationOutput output,
            Path modelPath,
            DirectPromptPreparation prompt,
            Map<String, Object> metadata) {
        DirectGenerationOutput safeOutput = DirectGenerationOutput.orEmpty(output);
        return new DirectInferenceEngine.DirectGenerationTrace(
                DirectInferenceResponses.finalResponse(request, safeOutput, modelPath, prompt.length(), metadata),
                copy(prompt.ids()),
                copy(safeOutput.generatedTokenIds()));
    }

    static DirectInferenceEngine.DirectConversationTrace conversationTrace(
            DirectGenerationRequestContext request,
            DirectGenerationOutput output,
            Path modelPath,
            DirectPromptPreparation prompt,
            KVCacheManager.KVCacheSession session,
            Map<String, Object> metadata) {
        DirectGenerationOutput safeOutput = DirectGenerationOutput.orEmpty(output);
        return conversationTrace(
                DirectInferenceResponses.finalResponse(request, safeOutput, modelPath, prompt.length(), metadata),
                prompt.ids(),
                safeOutput,
                session);
    }

    static DirectInferenceEngine.DirectConversationTrace streamConversationTrace(
            DirectGenerationRequestContext request,
            DirectGenerationOutput output,
            Path modelPath,
            DirectPromptPreparation prompt,
            KVCacheManager.KVCacheSession session,
            Map<String, Object> metadata) {
        DirectGenerationOutput safeOutput = DirectGenerationOutput.orEmpty(output);
        return conversationTrace(
                DirectInferenceResponses.finalResponse(request, "", safeOutput, modelPath, prompt.length(), metadata),
                prompt.ids(),
                safeOutput,
                session);
    }

    static DirectInferenceEngine.DirectConversationTrace continuationTrace(
            DirectGenerationRequestContext request,
            DirectGenerationOutput output,
            Path modelPath,
            DirectConversationContinuationPlan continuation,
            KVCacheManager.KVCacheSession session) {
        DirectGenerationOutput safeOutput = DirectGenerationOutput.orEmpty(output);
        long[] inputIds = continuation.fullInputIds();
        return conversationTrace(
                DirectInferenceResponses.finalResponse(
                        request, safeOutput, modelPath, inputIds.length, continuation.retainedKvMetadata()),
                inputIds,
                safeOutput,
                session);
    }

    static DirectInferenceEngine.DirectConversationTrace streamContinuationTrace(
            DirectGenerationRequestContext request,
            DirectGenerationOutput output,
            Path modelPath,
            DirectConversationContinuationPlan continuation,
            KVCacheManager.KVCacheSession session) {
        DirectGenerationOutput safeOutput = DirectGenerationOutput.orEmpty(output);
        long[] inputIds = continuation.fullInputIds();
        return conversationTrace(
                DirectInferenceResponses.finalResponse(
                        request, "", safeOutput, modelPath, inputIds.length, continuation.retainedKvMetadata()),
                inputIds,
                safeOutput,
                session);
    }

    private static DirectInferenceEngine.DirectConversationTrace conversationTrace(
            InferenceResponse response,
            long[] inputIds,
            DirectGenerationOutput output,
            KVCacheManager.KVCacheSession session) {
        return new DirectInferenceEngine.DirectConversationTrace(
                response,
                copy(inputIds),
                copy(output.generatedTokenIds()),
                session);
    }

    private static long[] copy(long[] values) {
        return values == null ? new long[0] : values.clone();
    }
}
