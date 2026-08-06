package tech.kayys.gollek.onnx.runner;

import java.util.Objects;

final class OnnxTextGenerationProgress {

    private long consumedTokens;
    private OnnxTextFinishReason finishReason = OnnxTextFinishReason.LENGTH;

    long consumedTokens() {
        return consumedTokens;
    }

    OnnxTextFinishReason finishReason() {
        return finishReason;
    }

    void advance(long sequenceLength) {
        if (sequenceLength <= 0L) {
            throw new IllegalArgumentException("sequenceLength must be > 0");
        }
        consumedTokens += sequenceLength;
    }

    void cancel() {
        finishReason = OnnxTextFinishReason.CANCELLED;
    }

    boolean apply(OnnxTextTokenDecision tokenDecision) {
        Objects.requireNonNull(tokenDecision, "tokenDecision");
        if (tokenDecision.finished()) {
            finishReason = tokenDecision.finishReason();
            return true;
        }
        return false;
    }
}
