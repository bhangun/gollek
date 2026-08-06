package tech.kayys.gollek.onnx.runner;

import java.util.Objects;

record OnnxTextGenerationResult(
        String requestId,
        String content,
        int inputTokens,
        int outputTokens,
        long durationMs,
        OnnxInferenceProfile profile,
        boolean fallback,
        OnnxTextFinishReason finishReason) {

    OnnxTextGenerationResult {
        finishReason = Objects.requireNonNull(finishReason, "finishReason");
    }
}
