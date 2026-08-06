package tech.kayys.gollek.onnx.runner;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

final class OnnxInputIdsScratch {

    private final MemorySegment values;
    private final MemorySegment latestValue;
    private final int capacity;
    private int filledPrefix;

    private OnnxInputIdsScratch(MemorySegment values, MemorySegment latestValue, int capacity) {
        this.values = values;
        this.latestValue = latestValue;
        this.capacity = capacity;
    }

    static OnnxInputIdsScratch allocate(Arena arena, int capacity) {
        Objects.requireNonNull(arena, "arena");
        if (capacity <= 0) {
            throw new IllegalArgumentException("Input id capacity must be positive: " + capacity);
        }
        return new OnnxInputIdsScratch(
                arena.allocate((long) capacity * Long.BYTES, Long.BYTES),
                arena.allocate(Long.BYTES, Long.BYTES),
                capacity);
    }

    MemorySegment prefix(OnnxTokenHistory history, int count) {
        return prefixView(history, count).asSizedSegment();
    }

    OnnxTensorDataView prefixView(OnnxTokenHistory history, int count) {
        Objects.requireNonNull(history, "history");
        validateCount(history, count);
        fillPrefix(history, count);
        return OnnxTensorDataView.int64Elements(values, count);
    }

    MemorySegment last(OnnxTokenHistory history) {
        return lastView(history).data();
    }

    OnnxTensorDataView lastView(OnnxTokenHistory history) {
        Objects.requireNonNull(history, "history");
        int size = history.size();
        if (size <= 0) {
            throw new IllegalStateException("Token history is empty");
        }
        history.writeLastAsInt64(latestValue);
        return OnnxTensorDataView.int64Elements(latestValue, 1);
    }

    int filledPrefix() {
        return filledPrefix;
    }

    void reset() {
        filledPrefix = 0;
    }

    private void fillPrefix(OnnxTokenHistory history, int count) {
        for (int i = filledPrefix; i < count; i++) {
            values.setAtIndex(ValueLayout.JAVA_LONG, i, history.tokenAt(i));
        }
        filledPrefix = Math.max(filledPrefix, count);
    }

    private void validateCount(OnnxTokenHistory history, int count) {
        if (count <= 0 || count > capacity || count > history.size()) {
            throw new IllegalArgumentException("Input id count " + count
                    + " is outside capacity " + capacity + " and history size " + history.size());
        }
    }
}
