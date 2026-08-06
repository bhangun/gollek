package tech.kayys.gollek.onnx.runner;

import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.function.Consumer;

final class OnnxLengthTensorCache {

    private final long[] lengths;
    private final MemorySegment[] tensors;
    private int size;
    private int hits;
    private int misses;
    private int evictions;

    OnnxLengthTensorCache(int maxEntries) {
        if (maxEntries < 0) {
            throw new IllegalArgumentException("maxEntries must be >= 0");
        }
        this.lengths = new long[maxEntries];
        this.tensors = new MemorySegment[maxEntries];
    }

    MemorySegment get(long length) {
        if (length <= 0L) {
            throw new IllegalArgumentException("length must be positive: " + length);
        }
        int index = indexOf(length);
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

    boolean retain(long length, MemorySegment value, Consumer<MemorySegment> releaser) {
        if (length <= 0L) {
            throw new IllegalArgumentException("length must be positive: " + length);
        }
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(releaser, "releaser");
        if (tensors.length == 0) {
            return false;
        }
        int existing = indexOf(length);
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
            lengths[i] = 0L;
        }
        size = 0;
        resetStats();
    }

    private int indexOf(long length) {
        for (int i = 0; i < size; i++) {
            if (lengths[i] == length) {
                return i;
            }
        }
        return -1;
    }

    private void moveToFront(int index) {
        if (index <= 0) {
            return;
        }
        long length = lengths[index];
        MemorySegment value = tensors[index];
        shiftRight(0, index);
        lengths[0] = length;
        tensors[0] = value;
    }

    private void shiftRight(int fromInclusive, int toExclusive) {
        for (int i = toExclusive; i > fromInclusive; i--) {
            lengths[i] = lengths[i - 1];
            tensors[i] = tensors[i - 1];
        }
    }
}
