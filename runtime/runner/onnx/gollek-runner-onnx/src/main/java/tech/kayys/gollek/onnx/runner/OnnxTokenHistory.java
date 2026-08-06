package tech.kayys.gollek.onnx.runner;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.Objects;

final class OnnxTokenHistory {

    private int[] tokens;
    private int size;

    private OnnxTokenHistory(int[] tokens, int size) {
        this.tokens = tokens;
        this.size = size;
    }

    static OnnxTokenHistory from(int[] prompt) {
        return from(prompt, 0);
    }

    static OnnxTokenHistory from(int[] prompt, int expectedGeneratedTokens) {
        Objects.requireNonNull(prompt, "prompt");
        if (expectedGeneratedTokens < 0) {
            throw new IllegalArgumentException("Expected generated token count must be non-negative: "
                    + expectedGeneratedTokens);
        }
        int capacity = capacityFor(prompt.length, expectedGeneratedTokens);
        int[] tokens = Arrays.copyOf(prompt, capacity);
        return new OnnxTokenHistory(tokens, prompt.length);
    }

    static OnnxTokenHistory allocate(int capacity) {
        if (capacity < 0) {
            throw new IllegalArgumentException("Token history capacity must be non-negative: " + capacity);
        }
        return new OnnxTokenHistory(new int[Math.max(8, capacity)], 0);
    }

    OnnxTokenHistory resetFrom(int[] prompt, int expectedGeneratedTokens) {
        Objects.requireNonNull(prompt, "prompt");
        ensureCapacity(capacityFor(prompt.length, expectedGeneratedTokens));
        System.arraycopy(prompt, 0, tokens, 0, prompt.length);
        size = prompt.length;
        return this;
    }

    int size() {
        return size;
    }

    int capacity() {
        return tokens.length;
    }

    int last() {
        if (size == 0) {
            throw new IllegalStateException("Token history is empty");
        }
        return tokens[size - 1];
    }

    int tokenAt(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Token index " + index + " is outside size " + size);
        }
        return tokens[index];
    }

    void append(int tokenId) {
        ensureCapacity(size + 1);
        tokens[size++] = tokenId;
    }

    void writeAllAsInt64(MemorySegment target) {
        writePrefixAsInt64(target, size);
    }

    void writeLastAsInt64(MemorySegment target) {
        target.setAtIndex(ValueLayout.JAVA_LONG, 0, last());
    }

    void writePrefixAsInt64(MemorySegment target, int count) {
        Objects.requireNonNull(target, "target");
        if (count < 0 || count > size) {
            throw new IllegalArgumentException("Prefix count " + count
                    + " is outside token history size " + size);
        }
        long requiredBytes = (long) count * Long.BYTES;
        if (target.byteSize() < requiredBytes) {
            throw new IllegalArgumentException("Target segment is too small: required "
                    + requiredBytes + " bytes, found " + target.byteSize());
        }
        for (int i = 0; i < count; i++) {
            target.setAtIndex(ValueLayout.JAVA_LONG, i, tokens[i]);
        }
    }

    private void ensureCapacity(int targetSize) {
        if (targetSize <= tokens.length) {
            return;
        }
        int next = tokens.length + (tokens.length >> 1) + 1;
        tokens = Arrays.copyOf(tokens, Math.max(next, targetSize));
    }

    private static int capacityFor(int promptLength, int expectedGeneratedTokens) {
        if (expectedGeneratedTokens < 0) {
            throw new IllegalArgumentException("Expected generated token count must be non-negative: "
                    + expectedGeneratedTokens);
        }
        long targetCapacity = (long) promptLength + expectedGeneratedTokens;
        if (targetCapacity > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Token history capacity is too large: " + targetCapacity);
        }
        return Math.max(8, (int) targetCapacity);
    }
}
