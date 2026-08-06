package tech.kayys.gollek.onnx.runner;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Supplier;

final class OnnxRepeatedInputValue {

    private final int repeatCount;
    private final Supplier<MemorySegment> valueFactory;
    private final Consumer<MemorySegment> releaser;
    private MemorySegment value;

    private OnnxRepeatedInputValue(
            int repeatCount,
            Supplier<MemorySegment> valueFactory,
            Consumer<MemorySegment> releaser) {
        this.repeatCount = repeatCount;
        this.valueFactory = valueFactory;
        this.releaser = releaser;
    }

    static OnnxRepeatedInputValue create(
            int repeatCount,
            Supplier<MemorySegment> valueFactory,
            Consumer<MemorySegment> releaser) {
        if (repeatCount < 0) {
            throw new IllegalArgumentException("Repeated input count must be non-negative: " + repeatCount);
        }
        Objects.requireNonNull(valueFactory, "valueFactory");
        Objects.requireNonNull(releaser, "releaser");
        return new OnnxRepeatedInputValue(repeatCount, valueFactory, releaser);
    }

    int appendTo(MemorySegment[] target, int offset) {
        Objects.requireNonNull(target, "target");
        if (offset < 0 || target.length - offset < repeatCount) {
            throw new IllegalArgumentException("Target input array length " + target.length
                    + " cannot accept " + repeatCount + " repeated values at offset " + offset);
        }
        if (repeatCount == 0) {
            return offset;
        }
        MemorySegment repeated = value();
        Arrays.fill(target, offset, offset + repeatCount, repeated);
        return offset + repeatCount;
    }

    void release() {
        if (value == null) {
            return;
        }
        releaser.accept(value);
        value = null;
    }

    private MemorySegment value() {
        if (value == null) {
            value = Objects.requireNonNull(valueFactory.get(), "valueFactory returned null");
        }
        return value;
    }
}
