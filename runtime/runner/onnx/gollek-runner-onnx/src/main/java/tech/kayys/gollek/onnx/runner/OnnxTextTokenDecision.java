package tech.kayys.gollek.onnx.runner;

import java.util.Objects;

record OnnxTextTokenDecision(boolean finished, OnnxTextFinishReason finishReason) {

    private static final OnnxTextTokenDecision CONTINUE =
            new OnnxTextTokenDecision(false, OnnxTextFinishReason.LENGTH);
    private static final OnnxTextTokenDecision[] STOP_BY_REASON = stopDecisions();

    static OnnxTextTokenDecision continueGeneration() {
        return CONTINUE;
    }

    static OnnxTextTokenDecision stop(OnnxTextFinishReason finishReason) {
        Objects.requireNonNull(finishReason, "finishReason");
        return STOP_BY_REASON[finishReason.ordinal()];
    }

    OnnxTextTokenDecision {
        finishReason = Objects.requireNonNull(finishReason, "finishReason");
    }

    private static OnnxTextTokenDecision[] stopDecisions() {
        OnnxTextFinishReason[] reasons = OnnxTextFinishReason.values();
        OnnxTextTokenDecision[] decisions = new OnnxTextTokenDecision[reasons.length];
        for (OnnxTextFinishReason reason : reasons) {
            decisions[reason.ordinal()] = new OnnxTextTokenDecision(true, reason);
        }
        return decisions;
    }
}
