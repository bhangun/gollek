/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import static tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler.maybePrintProfileSummary;
import static tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler.putGenerationBenchMetadata;

import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;

final class DirectInferenceResponses {
    static final String ENGINE_BACKEND = "accelerate-safetensor";

    private DirectInferenceResponses() {
    }

    static InferenceResponse finalResponse(DirectGenerationRequestContext request, DirectGenerationOutput output,
            Path modelPath, int inputTokens, Map<String, Object> extraMetadata) {
        DirectGenerationOutput safeOutput = DirectGenerationOutput.orEmpty(output);
        return finalResponse(request, safeOutput.text(), safeOutput.generatedTokenCount(), modelPath, inputTokens,
                extraMetadata);
    }

    static InferenceResponse finalResponse(DirectGenerationRequestContext request, String content,
            DirectGenerationOutput output, Path modelPath, int inputTokens, Map<String, Object> extraMetadata) {
        DirectGenerationOutput safeOutput = DirectGenerationOutput.orEmpty(output);
        return finalResponse(request, content, safeOutput.generatedTokenCount(), modelPath, inputTokens, extraMetadata);
    }

    static InferenceResponse finalBenchResponse(DirectGenerationRequestContext request, DirectGenerationOutput output,
            Path modelPath, int inputTokens, Map<String, Object> extraMetadata) {
        DirectGenerationOutput safeOutput = DirectGenerationOutput.orEmpty(output);
        return finalBenchResponse(request, safeOutput.text(), safeOutput.completionTokens(), modelPath, inputTokens,
                extraMetadata);
    }

    static InferenceResponse finalBenchResponse(DirectGenerationRequestContext request, String content,
            DirectGenerationOutput output, Path modelPath, int inputTokens, Map<String, Object> extraMetadata) {
        DirectGenerationOutput safeOutput = DirectGenerationOutput.orEmpty(output);
        return finalBenchResponse(request, content, safeOutput.completionTokens(), modelPath, inputTokens,
                extraMetadata);
    }

    static InferenceResponse streamDelta(DirectGenerationRequestContext request, String delta, Path modelPath,
            int inputTokens, Map<String, Object> extraMetadata) {
        return streamDelta(request.requestId(), delta, modelPath, inputTokens, extraMetadata);
    }

    private static InferenceResponse finalResponse(String requestId, String content, Path modelPath,
            int inputTokens, int outputTokens, Instant startedAt, InferenceProfile profile, String profileBackend,
            Map<String, Object> extraMetadata) {
        InferenceResponse.Builder builder = finalBuilder(requestId, content, modelPath, inputTokens, outputTokens,
                startedAt);
        applyMetadata(builder, extraMetadata);
        if (profile != null) {
            profile.metadata(profileBackend, inputTokens, outputTokens).forEach(builder::metadata);
            maybePrintProfileSummary(profile, profileBackend);
        }
        return builder.build();
    }

    private static InferenceResponse finalResponse(DirectGenerationRequestContext request, String content,
            int outputTokens, Path modelPath, int inputTokens, Map<String, Object> extraMetadata) {
        return finalResponse(request.requestId(), content, modelPath, inputTokens, outputTokens,
                request.startedAt(), request.profile(), request.backend(), extraMetadata);
    }

    private static InferenceResponse finalBenchResponse(DirectGenerationRequestContext request, String content,
            int outputTokens, Path modelPath, int inputTokens, Map<String, Object> extraMetadata) {
        return finalBenchResponse(request.requestId(), content, modelPath, inputTokens, outputTokens,
                request.startedAt(), request.profile(), request.backend(), request.benchTimings(), extraMetadata);
    }

    private static InferenceResponse finalBenchResponse(String requestId, String content, Path modelPath,
            int inputTokens, int outputTokens, Instant startedAt, InferenceProfile profile, String profileBackend,
            BenchTimings timings, Map<String, Object> extraMetadata) {
        InferenceResponse.Builder builder = finalBuilder(requestId, content, modelPath, inputTokens, outputTokens,
                startedAt);
        applyMetadata(builder, extraMetadata);

        Map<String, Object> benchMetadata = new LinkedHashMap<>();
        if (profile != null) {
            benchMetadata.putAll(profile.metadata(profileBackend, inputTokens, outputTokens));
            maybePrintProfileSummary(profile, profileBackend);
        }
        if (timings != null) {
            putGenerationBenchMetadata(benchMetadata, inputTokens, outputTokens,
                    timings.sessionAllocateNanos(), timings.prefillNanos(), timings.decodeNanos(),
                    timings.samplingNanos(), timings.firstTokenNanos(), timings.decodeSteps());
        }
        benchMetadata.forEach(builder::metadata);
        return builder.build();
    }

    private static InferenceResponse streamDelta(String requestId, String delta, Path modelPath, int inputTokens,
            Map<String, Object> extraMetadata) {
        InferenceResponse.Builder builder = InferenceResponse.builder()
                .requestId(requestId)
                .content(delta)
                .model(modelName(modelPath))
                .inputTokens(inputTokens)
                .metadata("backend", ENGINE_BACKEND);
        applyMetadata(builder, extraMetadata);
        return builder.build();
    }

    private static InferenceResponse.Builder finalBuilder(String requestId, String content, Path modelPath,
            int inputTokens, int outputTokens, Instant startedAt) {
        return InferenceResponse.builder()
                .requestId(requestId)
                .content(content)
                .model(modelName(modelPath))
                .inputTokens(inputTokens)
                .outputTokens(outputTokens)
                .durationMs(Duration.between(startedAt, Instant.now()).toMillis())
                .finishReason(InferenceResponse.FinishReason.STOP)
                .metadata("backend", ENGINE_BACKEND);
    }

    private static void applyMetadata(InferenceResponse.Builder builder, Map<String, Object> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return;
        }
        metadata.forEach(builder::metadata);
    }

    private static String modelName(Path modelPath) {
        return modelPath.getFileName().toString();
    }

    record BenchTimings(long sessionAllocateNanos, long prefillNanos, long decodeNanos, long samplingNanos,
            long firstTokenNanos, int decodeSteps) {
    }
}
