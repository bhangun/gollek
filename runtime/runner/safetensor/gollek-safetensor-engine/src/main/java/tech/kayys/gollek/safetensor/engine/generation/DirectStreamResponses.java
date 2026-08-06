/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Bridges decoded streaming deltas into inference response chunks.
 */
final class DirectStreamResponses {

    private DirectStreamResponses() {
    }

    static Consumer<String> deltaConsumer(
            Consumer<InferenceResponse> responseSink,
            DirectGenerationRequestContext request,
            Path modelPath,
            DirectPromptPreparation prompt,
            Map<String, Object> metadata) {
        return deltaConsumer(responseSink, request, modelPath, prompt.length(), metadata);
    }

    static Consumer<String> deltaConsumer(
            Consumer<InferenceResponse> responseSink,
            DirectGenerationRequestContext request,
            Path modelPath,
            int inputTokens,
            Map<String, Object> metadata) {
        Objects.requireNonNull(responseSink, "responseSink");
        return delta -> responseSink.accept(
                DirectInferenceResponses.streamDelta(request, delta, modelPath, inputTokens, metadata));
    }
}
