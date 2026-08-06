package tech.kayys.gollek.onnx.runner;

import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.util.Arrays;
import java.util.Objects;

final class OnnxGeneratedTokens {

    private int[] tokens;
    private long[] decodeScratch;
    private int size;

    private OnnxGeneratedTokens(int capacity) {
        this.tokens = new int[Math.max(1, capacity)];
    }

    static OnnxGeneratedTokens allocate(int capacity) {
        if (capacity < 0) {
            throw new IllegalArgumentException("Generated token capacity must be non-negative: " + capacity);
        }
        return new OnnxGeneratedTokens(capacity);
    }

    OnnxGeneratedTokens reset(int expectedCapacity) {
        if (expectedCapacity < 0) {
            throw new IllegalArgumentException("Generated token capacity must be non-negative: " + expectedCapacity);
        }
        ensureCapacity(expectedCapacity);
        size = 0;
        return this;
    }

    int size() {
        return size;
    }

    boolean isEmpty() {
        return size == 0;
    }

    void append(int tokenId) {
        ensureCapacity(size + 1);
        tokens[size++] = tokenId;
    }

    long[] toLongArray() {
        long[] values = new long[size];
        for (int i = 0; i < size; i++) {
            values[i] = tokens[i];
        }
        return values;
    }

    String decodeWith(Tokenizer tokenizer, DecodeOptions options) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(options, "options");
        if (size == 0) {
            return "";
        }
        long[] scratch = decodeScratch();
        for (int i = 0; i < size; i++) {
            scratch[i] = tokens[i];
        }
        return tokenizer.decode(scratch, 0, size, options);
    }

    String decodeEach(TokenDecoder decoder) {
        StringBuilder output = new StringBuilder();
        for (int i = 0; i < size; i++) {
            output.append(decoder.decode(tokens[i]));
        }
        return output.toString();
    }

    private void ensureCapacity(int targetSize) {
        if (targetSize <= tokens.length) {
            return;
        }
        int next = tokens.length + (tokens.length >> 1) + 1;
        tokens = Arrays.copyOf(tokens, Math.max(next, targetSize));
    }

    private long[] decodeScratch() {
        int required = Math.max(1, tokens.length);
        if (decodeScratch == null || decodeScratch.length < required) {
            decodeScratch = new long[required];
        }
        return decodeScratch;
    }

    @FunctionalInterface
    interface TokenDecoder {
        String decode(int tokenId);
    }
}
