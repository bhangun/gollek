/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class DirectGenerationTracesTest {

    private static final Path MODEL_PATH = Path.of("/tmp/trace-model");

    @AfterEach
    void clearActiveProfile() {
        DirectInferenceProfiler.clearProfile();
    }

    @Test
    void generationTraceCopiesPromptAndGeneratedTokenIds() {
        long[] inputIds = new long[] { 7L, 8L };
        long[] generatedIds = new long[] { 9L };
        DirectPromptPreparation prompt = pretokenized(inputIds);
        DirectGenerationOutput output = new DirectGenerationOutput("hello", generatedIds, 1);

        DirectInferenceEngine.DirectGenerationTrace trace = DirectGenerationTraces.generationTrace(
                DirectGenerationRequestContext.sync(null),
                output,
                MODEL_PATH,
                prompt,
                Map.of("source", "trace"));

        assertEquals("hello", trace.response().getContent());
        assertEquals(2, trace.response().getInputTokens());
        assertEquals(1, trace.response().getOutputTokens());
        assertEquals("trace", trace.response().getMetadata().get("source"));
        assertArrayEquals(new long[] { 7L, 8L }, trace.inputIds());
        assertArrayEquals(new long[] { 9L }, trace.generatedTokenIds());

        inputIds[0] = 99L;
        generatedIds[0] = 100L;
        assertArrayEquals(new long[] { 7L, 8L }, trace.inputIds());
        assertArrayEquals(new long[] { 9L }, trace.generatedTokenIds());
    }

    @Test
    void streamConversationTraceUsesEmptyFinalContent() {
        DirectPromptPreparation prompt = pretokenized(new long[] { 1L, 2L, 3L });
        DirectGenerationOutput output = new DirectGenerationOutput("already-streamed", new long[] { 4L, 5L }, 2);

        DirectInferenceEngine.DirectConversationTrace trace = DirectGenerationTraces.streamConversationTrace(
                DirectGenerationRequestContext.stream(null),
                output,
                MODEL_PATH,
                prompt,
                null,
                DirectGenerationMetadata.retainedPretokenizedConversation());

        assertEquals("", trace.response().getContent());
        assertEquals(3, trace.response().getInputTokens());
        assertEquals(2, trace.response().getOutputTokens());
        assertEquals(DirectGenerationMetadata.PRETOKENIZED,
                trace.response().getMetadata().get(DirectGenerationMetadata.PROMPT_TOKEN_SOURCE));
        assertArrayEquals(new long[] { 1L, 2L, 3L }, trace.inputIds());
        assertArrayEquals(new long[] { 4L, 5L }, trace.generatedTokenIds());
    }

    @Test
    void continuationTraceUsesValidatedContinuationInput() {
        DirectConversationContinuationPlan continuation =
                DirectConversationContinuationPlan.resolve(new long[] { 10L, 11L, 12L }, 2, 2, null);
        DirectGenerationOutput output = new DirectGenerationOutput("next", new long[] { 13L }, 1);

        DirectInferenceEngine.DirectConversationTrace trace = DirectGenerationTraces.continuationTrace(
                DirectGenerationRequestContext.sync(null), output, MODEL_PATH, continuation, null);

        assertEquals("next", trace.response().getContent());
        assertEquals(3, trace.response().getInputTokens());
        assertEquals(1, trace.response().getOutputTokens());
        assertEquals(DirectGenerationMetadata.CONVERSATION_DELTA,
                trace.response().getMetadata().get(DirectGenerationMetadata.PROMPT_TOKEN_SOURCE));
        assertArrayEquals(new long[] { 10L, 11L, 12L }, trace.inputIds());
        assertArrayEquals(new long[] { 13L }, trace.generatedTokenIds());
    }

    private static DirectPromptPreparation pretokenized(long[] inputIds) {
        return DirectPromptPreparation.pretokenized(
                null, new ModelConfig(), inputIds, DirectPromptPreparation.DebugOptions.none());
    }
}
