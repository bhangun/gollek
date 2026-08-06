package tech.kayys.gollek.onnx.runner;

import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.function.Consumer;

final class OnnxRunOutputLifecycle {

    private final OnnxRunOutputValues outputValues;
    private final OnnxPastKvState pastKvState;
    private final boolean capturePastKv;
    private final Consumer<MemorySegment> releaser;

    private OnnxRunOutputLifecycle(
            OnnxRunOutputValues outputValues,
            OnnxPastKvState pastKvState,
            boolean capturePastKv,
            Consumer<MemorySegment> releaser) {
        this.outputValues = Objects.requireNonNull(outputValues, "outputValues");
        this.pastKvState = Objects.requireNonNull(pastKvState, "pastKvState");
        this.capturePastKv = capturePastKv;
        this.releaser = Objects.requireNonNull(releaser, "releaser");
    }

    static OnnxRunOutputLifecycle create(
            OnnxRunOutputValues outputValues,
            OnnxPastKvState pastKvState,
            boolean capturePastKv,
            Consumer<MemorySegment> releaser) {
        return new OnnxRunOutputLifecycle(outputValues, pastKvState, capturePastKv, releaser);
    }

    void capturePresentAndReleaseLogits() {
        outputValues.capturePresentTo(pastKvState, capturePastKv, releaser);
        outputValues.releaseLogits(releaser);
    }

    void releaseUncapturedAndCurrentPastKv() {
        outputValues.releaseUncaptured(releaser);
        pastKvState.releaseCurrent(releaser);
    }
}
