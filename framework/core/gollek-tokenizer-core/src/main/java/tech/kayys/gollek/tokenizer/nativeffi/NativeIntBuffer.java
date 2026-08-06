package tech.kayys.gollek.tokenizer.nativeffi;

import java.lang.foreign.*;

import static java.lang.foreign.ValueLayout.JAVA_INT;

public class NativeIntBuffer {

    private final MemorySegment segment;
    private final int capacity;

    public NativeIntBuffer(Arena arena, int capacity) {
        this.capacity = capacity;
        this.segment = arena.allocate(JAVA_INT, capacity);
    }

    public MemorySegment segment() {
        return segment;
    }

    public int capacity() {
        return capacity;
    }
}