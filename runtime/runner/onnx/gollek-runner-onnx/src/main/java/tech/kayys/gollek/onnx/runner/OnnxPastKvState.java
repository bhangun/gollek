package tech.kayys.gollek.onnx.runner;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.Consumer;

final class OnnxPastKvState {

    private final int valueCount;
    private final MemorySegment[] firstBuffer;
    private MemorySegment[] spareBuffer;
    private MemorySegment[] current;

    private OnnxPastKvState(int valueCount) {
        this.valueCount = valueCount;
        this.firstBuffer = new MemorySegment[valueCount];
        this.spareBuffer = new MemorySegment[valueCount];
    }

    static OnnxPastKvState allocate(int valueCount) {
        if (valueCount < 0) {
            throw new IllegalArgumentException("Past KV value count must be non-negative: " + valueCount);
        }
        return new OnnxPastKvState(valueCount);
    }

    boolean hasCurrent() {
        return current != null;
    }

    MemorySegment[] current() {
        return current;
    }

    void capturePresentOutputs(MemorySegment[] outputs, int outputOffset, Consumer<MemorySegment> releaser) {
        Objects.requireNonNull(outputs, "outputs");
        Objects.requireNonNull(releaser, "releaser");
        if (valueCount == 0) {
            return;
        }
        if (outputOffset < 0 || outputs.length - outputOffset < valueCount) {
            throw new IllegalArgumentException("Output count " + outputs.length
                    + " cannot provide " + valueCount + " past KV values from offset " + outputOffset);
        }

        MemorySegment[] target = current == null ? firstBuffer : spareBuffer;
        System.arraycopy(outputs, outputOffset, target, 0, valueCount);

        MemorySegment[] previous = current;
        current = target;
        if (previous != null) {
            releaseAndClear(previous, releaser);
            spareBuffer = previous;
        }
    }

    void releaseCurrent(Consumer<MemorySegment> releaser) {
        Objects.requireNonNull(releaser, "releaser");
        if (current == null) {
            return;
        }
        releaseAndClear(current, releaser);
        current = null;
    }

    private static void releaseAndClear(MemorySegment[] values, Consumer<MemorySegment> releaser) {
        for (MemorySegment value : values) {
            releaser.accept(value);
        }
        Arrays.fill(values, null);
    }
}
