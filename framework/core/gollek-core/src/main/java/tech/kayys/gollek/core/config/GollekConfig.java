package tech.kayys.gollek.core.config;



import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

public final class GollekConfig {
    private static GollekConfig instance;

    private final long minMemoryMB;
    private final int maxSequenceLength;
    private final int maxBatchSize;
    private final boolean enableFlashAttention;

    private GollekConfig(Builder builder) {
        this.minMemoryMB = builder.minMemoryMB;
        this.maxSequenceLength = builder.maxSequenceLength;
        this.maxBatchSize = builder.maxBatchSize;
        this.enableFlashAttention = builder.enableFlashAttention;

        validate();
    }

    public static GollekConfig getDefault() {
        if (instance == null) {
            instance = builder().build();
        }
        return instance;
    }

    private void validate() {
        long availableMB = Runtime.getRuntime().maxMemory() / (1024 * 1024);
        if (availableMB < minMemoryMB) {
            throw new IllegalStateException(
                    String.format("Insufficient memory: need %d MB, have %d MB",
                            minMemoryMB, availableMB));
        }

        if (maxSequenceLength <= 0) {
            throw new InferenceException(ErrorCode.VALIDATION_INVALID_FORMAT, "maxSequenceLength must be > 0");
        }

        if (maxBatchSize <= 0) {
            throw new InferenceException(ErrorCode.VALIDATION_INVALID_FORMAT, "maxBatchSize must be > 0");
        }
    }

    public long getMinMemoryMB() {
        return minMemoryMB;
    }

    public int getMaxSequenceLength() {
        return maxSequenceLength;
    }

    public int getMaxBatchSize() {
        return maxBatchSize;
    }

    public boolean isFlashAttentionEnabled() {
        return enableFlashAttention;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private long minMemoryMB = 512;
        private int maxSequenceLength = 2048;
        private int maxBatchSize = 32;
        private boolean enableFlashAttention = true;

        public Builder minMemoryMB(long mb) {
            this.minMemoryMB = mb;
            return this;
        }

        public Builder maxSequenceLength(int length) {
            this.maxSequenceLength = length;
            return this;
        }

        public Builder maxBatchSize(int size) {
            this.maxBatchSize = size;
            return this;
        }

        public Builder enableFlashAttention(boolean enable) {
            this.enableFlashAttention = enable;
            return this;
        }

        public GollekConfig build() {
            return new GollekConfig(this);
        }
    }
}