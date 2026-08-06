package tech.kayys.gollek.onnx.runner;

import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.function.Consumer;

final class OnnxRunOutputValues {

    private final MemorySegment[] values;
    private boolean runCompleted;
    private boolean logitsReleased;
    private boolean presentCaptured;

    private OnnxRunOutputValues(int outputCount) {
        this.values = new MemorySegment[outputCount];
    }

    static OnnxRunOutputValues allocate(int outputCount) {
        if (outputCount <= 0) {
            throw new IllegalArgumentException("Output count must be positive: " + outputCount);
        }
        return new OnnxRunOutputValues(outputCount);
    }

    MemorySegment[] values() {
        return values;
    }

    void prepareForRun() {
        runCompleted = false;
        logitsReleased = false;
        presentCaptured = false;
    }

    void markRunCompleted() {
        runCompleted = true;
    }

    MemorySegment logits() {
        ensureCompleted();
        return values[0];
    }

    void capturePresentTo(OnnxPastKvState state, boolean enabled, Consumer<MemorySegment> releaser) {
        Objects.requireNonNull(state, "state");
        Objects.requireNonNull(releaser, "releaser");
        ensureCompleted();
        if (enabled && values.length > 1) {
            state.capturePresentOutputs(values, 1, releaser);
        }
        presentCaptured = true;
    }

    void releaseLogits(Consumer<MemorySegment> releaser) {
        Objects.requireNonNull(releaser, "releaser");
        if (!runCompleted || logitsReleased) {
            return;
        }
        releaser.accept(values[0]);
        values[0] = MemorySegment.NULL;
        logitsReleased = true;
    }

    void releaseUncaptured(Consumer<MemorySegment> releaser) {
        Objects.requireNonNull(releaser, "releaser");
        if (!runCompleted) {
            return;
        }
        releaseLogits(releaser);
        if (!presentCaptured) {
            for (int i = 1; i < values.length; i++) {
                releaser.accept(values[i]);
                values[i] = MemorySegment.NULL;
            }
        }
        runCompleted = false;
        presentCaptured = false;
    }

    private void ensureCompleted() {
        if (!runCompleted) {
            throw new IllegalStateException("ONNX run outputs are not available yet");
        }
    }
}
