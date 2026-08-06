/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DirectStreamResponsesTest {

    private static final Path MODEL_PATH = Path.of("/tmp/stream-model");

    @AfterEach
    void clearActiveProfile() {
        DirectInferenceProfiler.clearProfile();
    }

    @Test
    void emitsDeltaResponseUsingExplicitInputTokens() {
        DirectGenerationRequestContext request = DirectGenerationRequestContext.stream(null);
        List<InferenceResponse> responses = new ArrayList<>();

        Consumer<String> deltaConsumer = DirectStreamResponses.deltaConsumer(
                responses::add, request, MODEL_PATH, 4, Map.of("source", "explicit"));

        deltaConsumer.accept("hi");

        assertEquals(1, responses.size());
        InferenceResponse response = responses.get(0);
        assertEquals(request.requestId(), response.getRequestId());
        assertEquals("hi", response.getContent());
        assertEquals("stream-model", response.getModel());
        assertEquals(4, response.getInputTokens());
        assertEquals("accelerate-safetensor", response.getMetadata().get("backend"));
        assertEquals("explicit", response.getMetadata().get("source"));
    }

    @Test
    void emitsDeltaResponseUsingPromptLength() {
        DirectPromptPreparation prompt = DirectPromptPreparation.pretokenized(
                null, new ModelConfig(), new long[] { 1L, 2L, 3L },
                DirectPromptPreparation.DebugOptions.none());
        List<InferenceResponse> responses = new ArrayList<>();

        DirectStreamResponses.deltaConsumer(
                responses::add,
                DirectGenerationRequestContext.stream(null),
                MODEL_PATH,
                prompt,
                DirectGenerationMetadata.pretokenizedPrompt())
                .accept("x");

        assertEquals(1, responses.size());
        assertEquals(3, responses.get(0).getInputTokens());
        assertEquals(DirectGenerationMetadata.PRETOKENIZED,
                responses.get(0).getMetadata().get(DirectGenerationMetadata.PROMPT_TOKEN_SOURCE));
    }

    @Test
    void rejectsMissingResponseSink() {
        assertThrows(NullPointerException.class, () -> DirectStreamResponses.deltaConsumer(
                null, DirectGenerationRequestContext.stream(null), MODEL_PATH, 1, null));
    }
}
