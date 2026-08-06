/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DirectInferenceResponsesTest {

    private static final Path MODEL_PATH = Path.of("/tmp/gollek-test-model");

    @AfterEach
    void clearActiveProfile() {
        DirectInferenceProfiler.clearProfile();
    }

    @Test
    void finalResponseUsesRequestContextAndGeneratedTokenIds() {
        DirectGenerationRequestContext request = DirectGenerationRequestContext.sync(null);
        DirectGenerationOutput output = new DirectGenerationOutput("hello", new long[] { 1L, 2L }, 99);

        InferenceResponse response = DirectInferenceResponses.finalResponse(
                request, output, MODEL_PATH, 3, Map.of("source", "test"));

        assertEquals(request.requestId(), response.getRequestId());
        assertEquals("hello", response.getContent());
        assertEquals("gollek-test-model", response.getModel());
        assertEquals(3, response.getInputTokens());
        assertEquals(2, response.getOutputTokens());
        assertEquals("accelerate-safetensor", response.getMetadata().get("backend"));
        assertEquals("test", response.getMetadata().get("source"));
    }

    @Test
    void finalBenchResponseUsesCompletionTokenCount() {
        DirectGenerationRequestContext request = DirectGenerationRequestContext.sync(null);
        DirectGenerationOutput output = new DirectGenerationOutput("bench", new long[0], 4);

        InferenceResponse response = DirectInferenceResponses.finalBenchResponse(
                request, "", output, MODEL_PATH, 5, null);

        assertEquals(request.requestId(), response.getRequestId());
        assertEquals("", response.getContent());
        assertEquals(5, response.getInputTokens());
        assertEquals(4, response.getOutputTokens());
        assertEquals("accelerate-safetensor", response.getMetadata().get("backend"));
    }

    @Test
    void streamDeltaUsesRequestContext() {
        DirectGenerationRequestContext request = DirectGenerationRequestContext.stream(null);

        InferenceResponse response = DirectInferenceResponses.streamDelta(
                request, "d", MODEL_PATH, 6, Map.of("source", "stream"));

        assertEquals(request.requestId(), response.getRequestId());
        assertEquals("d", response.getContent());
        assertEquals("gollek-test-model", response.getModel());
        assertEquals(6, response.getInputTokens());
        assertEquals("accelerate-safetensor", response.getMetadata().get("backend"));
        assertEquals("stream", response.getMetadata().get("source"));
    }
}
