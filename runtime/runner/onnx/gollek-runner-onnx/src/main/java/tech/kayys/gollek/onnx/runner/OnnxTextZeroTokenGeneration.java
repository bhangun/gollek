package tech.kayys.gollek.onnx.runner;

import java.util.Objects;

final class OnnxTextZeroTokenGeneration {

    private OnnxTextZeroTokenGeneration() {
    }

    static OnnxTextGenerationResult finish(OnnxTextGenerationSetup setup, long requestStartMillis) {
        Objects.requireNonNull(setup, "setup");
        return new OnnxTextGenerationResult(
                setup.requestId(),
                "",
                setup.promptLength(),
                0,
                Math.max(0L, System.currentTimeMillis() - requestStartMillis),
                setup.profile(),
                false,
                OnnxTextFinishReason.LENGTH);
    }
}
