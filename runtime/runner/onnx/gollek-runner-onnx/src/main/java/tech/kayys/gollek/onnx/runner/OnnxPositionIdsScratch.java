package tech.kayys.gollek.onnx.runner;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

final class OnnxPositionIdsScratch {

    private final MemorySegment values;
    private final MemorySegment latestValue;
    private final long capacity;
    private long filledPositions;

    private OnnxPositionIdsScratch(MemorySegment values, MemorySegment latestValue, long capacity) {
        this.values = values;
        this.latestValue = latestValue;
        this.capacity = capacity;
    }

    static OnnxPositionIdsScratch allocate(Arena arena, long capacity) {
        Objects.requireNonNull(arena, "arena");
        if (capacity <= 0L) {
            throw new IllegalArgumentException("Position id capacity must be positive: " + capacity);
        }
        return new OnnxPositionIdsScratch(
                arena.allocate(capacity * Long.BYTES, Long.BYTES),
                arena.allocate(Long.BYTES, Long.BYTES),
                capacity);
    }

    MemorySegment positions(long start, long length) {
        return positionsView(start, length).asSizedSegment();
    }

    OnnxTensorDataView positionsView(long start, long length) {
        if (start < 0L || length <= 0L || start > capacity || length > capacity - start) {
            throw new IllegalArgumentException("Position id range start=" + start + " length=" + length
                    + " is outside capacity " + capacity);
        }
        if (length == 1L) {
            latestValue.setAtIndex(ValueLayout.JAVA_LONG, 0, start);
            return OnnxTensorDataView.int64Elements(latestValue, 1);
        }
        long required = start + length;
        fillPositions(required);
        if (start == 0L) {
            return OnnxTensorDataView.int64Elements(values, length);
        }
        return OnnxTensorDataView.int64Elements(values.asSlice(start * Long.BYTES, length * Long.BYTES), length);
    }

    long filledPositions() {
        return filledPositions;
    }

    private void fillPositions(long required) {
        for (long i = filledPositions; i < required; i++) {
            values.setAtIndex(ValueLayout.JAVA_LONG, i, i);
        }
        filledPositions = Math.max(filledPositions, required);
    }
}
