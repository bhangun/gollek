package tech.kayys.gollek.onnx.runner;

import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.function.Consumer;

final class OnnxPositionIdsTensorCache {

    private final long[] starts;
    private final long[] lengths;
    private final MemorySegment[] tensors;
    private int size;
    private int hits;
    private int misses;
    private int evictions;

    OnnxPositionIdsTensorCache(int maxEntries) {
        if (maxEntries < 0) {
            throw new IllegalArgumentException("maxEntries must be >= 0");
        }
        this.starts = new long[maxEntries];
        this.lengths = new long[maxEntries];
        this.tensors = new MemorySegment[maxEntries];
    }

    MemorySegment get(long start, long length) {
        validateKey(start, length);
        int index = indexOf(start, length);
        if (index < 0) {
            misses++;
            return null;
        }
        hits++;
        MemorySegment value = tensors[index];
        moveToFront(index);
        return value;
    }

    int hits() {
        return hits;
    }

    int misses() {
        return misses;
    }

    int evictions() {
        return evictions;
    }

    boolean retain(long start, long length, MemorySegment value, Consumer<MemorySegment> releaser) {
        validateKey(start, length);
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(releaser, "releaser");
        if (tensors.length == 0) {
            return false;
        }
        int existing = indexOf(start, length);
        if (existing >= 0) {
            MemorySegment previous = tensors[existing];
            if (previous != value) {
                releaser.accept(previous);
            }
            tensors[existing] = value;
            moveToFront(existing);
            return true;
        }
        if (size == tensors.length) {
            int evictIndex = size - 1;
            releaser.accept(tensors[evictIndex]);
            evictions++;
        } else {
            size++;
        }
        shiftRight(0, size - 1);
        starts[0] = start;
        lengths[0] = length;
        tensors[0] = value;
        return true;
    }

    void resetStats() {
        hits = 0;
        misses = 0;
        evictions = 0;
    }

    void releaseAll(Consumer<MemorySegment> releaser) {
        Objects.requireNonNull(releaser, "releaser");
        for (int i = 0; i < size; i++) {
            releaser.accept(tensors[i]);
            tensors[i] = null;
            starts[i] = 0L;
            lengths[i] = 0L;
        }
        size = 0;
        resetStats();
    }

    private static void validateKey(long start, long length) {
        if (start < 0L) {
            throw new IllegalArgumentException("start must be non-negative: " + start);
        }
        if (length <= 0L) {
            throw new IllegalArgumentException("length must be positive: " + length);
        }
    }

    private int indexOf(long start, long length) {
        for (int i = 0; i < size; i++) {
            if (starts[i] == start && lengths[i] == length) {
                return i;
            }
        }
        return -1;
    }

    private void moveToFront(int index) {
        if (index <= 0) {
            return;
        }
        long start = starts[index];
        long length = lengths[index];
        MemorySegment value = tensors[index];
        shiftRight(0, index);
        starts[0] = start;
        lengths[0] = length;
        tensors[0] = value;
    }

    private void shiftRight(int fromInclusive, int toExclusive) {
        for (int i = toExclusive; i > fromInclusive; i--) {
            starts[i] = starts[i - 1];
            lengths[i] = lengths[i - 1];
            tensors[i] = tensors[i - 1];
        }
    }
}
