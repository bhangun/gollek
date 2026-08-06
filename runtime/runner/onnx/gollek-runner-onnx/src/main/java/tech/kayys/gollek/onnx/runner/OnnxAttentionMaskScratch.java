package tech.kayys.gollek.onnx.runner;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

final class OnnxAttentionMaskScratch {

    private final MemorySegment values;
    private final long capacity;
    private long filledOnes;

    private OnnxAttentionMaskScratch(MemorySegment values, long capacity) {
        this.values = values;
        this.capacity = capacity;
    }

    static OnnxAttentionMaskScratch allocate(Arena arena, long capacity) {
        Objects.requireNonNull(arena, "arena");
        if (capacity <= 0L) {
            throw new IllegalArgumentException("Attention mask capacity must be positive: " + capacity);
        }
        return new OnnxAttentionMaskScratch(arena.allocate(capacity * Long.BYTES, Long.BYTES), capacity);
    }

    MemorySegment ones(long length) {
        return onesView(length).asSizedSegment();
    }

    OnnxTensorDataView onesView(long length) {
        if (length <= 0L || length > capacity) {
            throw new IllegalArgumentException("Attention mask length " + length
                    + " is outside capacity " + capacity);
        }
        for (long i = filledOnes; i < length; i++) {
            values.setAtIndex(ValueLayout.JAVA_LONG, i, 1L);
        }
        filledOnes = Math.max(filledOnes, length);
        return OnnxTensorDataView.int64Elements(values, length);
    }

    long filledOnes() {
        return filledOnes;
    }
}
