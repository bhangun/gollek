package tech.kayys.gollek.onnx.runner;

import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.function.Consumer;

final class OnnxRunInputValues {

    private final MemorySegment[] values;
    private final MemorySegment[] ownedValues;
    private int valueCount;
    private int ownedCount;

    private OnnxRunInputValues(int inputCount, int ownedCapacity) {
        this.values = new MemorySegment[inputCount];
        this.ownedValues = new MemorySegment[ownedCapacity];
    }

    static OnnxRunInputValues allocate(int inputCount, int ownedCapacity) {
        if (inputCount < 0) {
            throw new IllegalArgumentException("Input count must be non-negative: " + inputCount);
        }
        if (ownedCapacity < 0) {
            throw new IllegalArgumentException("Owned input capacity must be non-negative: " + ownedCapacity);
        }
        if (ownedCapacity > inputCount) {
            throw new IllegalArgumentException("Owned input capacity " + ownedCapacity
                    + " cannot exceed input count " + inputCount);
        }
        return new OnnxRunInputValues(inputCount, ownedCapacity);
    }

    void reset() {
        valueCount = 0;
        ownedCount = 0;
    }

    void addOwned(MemorySegment value) {
        if (ownedCount >= ownedValues.length) {
            throw new IllegalStateException("Owned input capacity exceeded: " + ownedValues.length);
        }
        addBorrowed(value);
        ownedValues[ownedCount++] = value;
    }

    void addBorrowed(MemorySegment value) {
        Objects.requireNonNull(value, "value");
        if (valueCount >= values.length) {
            throw new IllegalStateException("ONNX input value capacity exceeded: " + values.length);
        }
        values[valueCount++] = value;
    }

    void addBorrowedAll(MemorySegment[] borrowedValues, int count) {
        Objects.requireNonNull(borrowedValues, "borrowedValues");
        validateBorrowedWindow(borrowedValues, count);
        for (int i = 0; i < count; i++) {
            if (borrowedValues[i] == null) {
                throw new NullPointerException("borrowedValues[" + i + "]");
            }
        }
        addBorrowedAllUnchecked(borrowedValues, count);
    }

    void addBorrowedAllUnchecked(MemorySegment[] borrowedValues, int count) {
        Objects.requireNonNull(borrowedValues, "borrowedValues");
        validateBorrowedWindow(borrowedValues, count);
        System.arraycopy(borrowedValues, 0, values, valueCount, count);
        valueCount += count;
    }

    void addRepeated(OnnxRepeatedInputValue repeatedValue) {
        Objects.requireNonNull(repeatedValue, "repeatedValue");
        valueCount = repeatedValue.appendTo(values, valueCount);
    }

    MemorySegment[] completeValues() {
        if (valueCount != values.length) {
            throw new IllegalStateException("ONNX input plan/value count mismatch: planned="
                    + values.length + " actual=" + valueCount);
        }
        return values;
    }

    int count() {
        return valueCount;
    }

    void releaseOwned(Consumer<MemorySegment> releaser) {
        Objects.requireNonNull(releaser, "releaser");
        for (int i = 0; i < ownedCount; i++) {
            releaser.accept(ownedValues[i]);
            ownedValues[i] = null;
        }
        ownedCount = 0;
    }

    private void validateBorrowedWindow(MemorySegment[] borrowedValues, int count) {
        if (count < 0 || count > borrowedValues.length) {
            throw new IllegalArgumentException("Borrowed input count " + count
                    + " is outside value array length " + borrowedValues.length);
        }
        if (values.length - valueCount < count) {
            throw new IllegalStateException("ONNX input value capacity exceeded: " + values.length);
        }
    }
}
