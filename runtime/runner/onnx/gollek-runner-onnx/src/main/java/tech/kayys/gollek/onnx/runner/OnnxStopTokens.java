package tech.kayys.gollek.onnx.runner;

import java.util.Arrays;

final class OnnxStopTokens {

    private final int[] tokenIds;

    private OnnxStopTokens(int[] tokenIds) {
        this.tokenIds = tokenIds;
    }

    static OnnxStopTokens of(int... tokenIds) {
        Builder builder = builder();
        if (tokenIds != null) {
            for (int tokenId : tokenIds) {
                builder.add(tokenId);
            }
        }
        return builder.build();
    }

    static Builder builder() {
        return new Builder();
    }

    static Builder builder(OnnxStopTokens base) {
        Builder builder = builder();
        if (base != null) {
            for (int tokenId : base.tokenIds) {
                builder.add(tokenId);
            }
        }
        return builder;
    }

    boolean contains(int tokenId) {
        for (int id : tokenIds) {
            if (id == tokenId) {
                return true;
            }
        }
        return false;
    }

    int size() {
        return tokenIds.length;
    }

    int[] toArray() {
        return Arrays.copyOf(tokenIds, tokenIds.length);
    }

    static final class Builder {

        private int[] tokenIds = new int[4];
        private int size;

        Builder add(int tokenId) {
            if (contains(tokenId)) {
                return this;
            }
            ensureCapacity(size + 1);
            tokenIds[size++] = tokenId;
            return this;
        }

        Builder addAll(int[] values) {
            if (values == null) {
                return this;
            }
            for (int value : values) {
                add(value);
            }
            return this;
        }

        OnnxStopTokens build() {
            return new OnnxStopTokens(Arrays.copyOf(tokenIds, size));
        }

        private boolean contains(int tokenId) {
            for (int i = 0; i < size; i++) {
                if (tokenIds[i] == tokenId) {
                    return true;
                }
            }
            return false;
        }

        private void ensureCapacity(int targetSize) {
            if (targetSize <= tokenIds.length) {
                return;
            }
            int next = tokenIds.length + (tokenIds.length >> 1) + 1;
            tokenIds = Arrays.copyOf(tokenIds, Math.max(next, targetSize));
        }
    }
}
