/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QuantStats.java
 * ───────────────────────
 * Quantization statistics and metrics.
 */
package tech.kayys.gollek.safetensor.quantization;

import java.time.Duration;
import java.time.Instant;
import java.util.Objects;

/**
 * Statistics and metrics from quantization process.
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class QuantStats {

    /**
     * Original model size in bytes.
     */
    private final long originalSizeBytes;

    /**
     * Quantized model size in bytes.
     */
    private final long quantizedSizeBytes;

    /**
     * Compression ratio (original / quantized).
     */
    private final double compressionRatio;

    /**
     * Quantization duration.
     */
    private final Duration duration;

    /**
     * Peak memory usage during quantization in bytes.
     */
    private final long peakMemoryBytes;

    /**
     * Number of tensors quantized.
     */
    private final int tensorCount;

    /**
     * Number of parameters quantized.
     */
    private final long paramCount;

    /**
     * Average quantization error (MSE).
     */
    private final double avgQuantErrorMse;

    /**
     * Maximum quantization error.
     */
    private final double maxQuantError;

    /**
     * Start time of quantization.
     */
    private final Instant startTime;

    /**
     * End time of quantization.
     */
    private final Instant endTime;

    /**
     * Quantization status.
     */
    private final Status status;

    /**
     * Error message if failed.
     */
    private final String errorMessage;

    private QuantStats(Builder builder) {
        this.originalSizeBytes = builder.originalSizeBytes;
        this.quantizedSizeBytes = builder.quantizedSizeBytes;
        this.compressionRatio = builder.quantizedSizeBytes > 0
                ? (double) builder.originalSizeBytes / builder.quantizedSizeBytes
                : 0.0;
        this.duration = builder.endTime != null && builder.startTime != null
                ? Duration.between(builder.startTime, builder.endTime)
                : Duration.ZERO;
        this.peakMemoryBytes = builder.peakMemoryBytes;
        this.tensorCount = builder.tensorCount;
        this.paramCount = builder.paramCount;
        this.avgQuantErrorMse = builder.avgQuantErrorMse;
        this.maxQuantError = builder.maxQuantError;
        this.startTime = builder.startTime;
        this.endTime = builder.endTime;
        this.status = builder.status;
        this.errorMessage = builder.errorMessage;
    }

    /**
     * Create builder for QuantStats.
     *
     * @return new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    public long getOriginalSizeBytes() {
        return originalSizeBytes;
    }

    public long getQuantizedSizeBytes() {
        return quantizedSizeBytes;
    }

    public double getCompressionRatio() {
        return compressionRatio;
    }

    public Duration getDuration() {
        return duration;
    }

    public long getPeakMemoryBytes() {
        return peakMemoryBytes;
    }

    public int getTensorCount() {
        return tensorCount;
    }

    public long getParamCount() {
        return paramCount;
    }

    public double getAvgQuantErrorMse() {
        return avgQuantErrorMse;
    }

    public double getMaxQuantError() {
        return maxQuantError;
    }

    public Instant getStartTime() {
        return startTime;
    }

    public Instant getEndTime() {
        return endTime;
    }

    public Status getStatus() {
        return status;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    /**
     * Get human-readable size string.
     *
     * @param bytes size in bytes
     * @return formatted size string
     */
    public static String formatSize(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        } else if (bytes < 1024 * 1024) {
            return String.format("%.2f KB", bytes / 1024.0);
        } else if (bytes < 1024 * 1024 * 1024) {
            return String.format("%.2f MB", bytes / (1024.0 * 1024.0));
        } else if (bytes < 1024L * 1024L * 1024L * 1024L) {
            return String.format("%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
        } else {
            return String.format("%.2f TB", bytes / (1024.0 * 1024.0 * 1024.0 * 1024.0));
        }
    }

    /**
     * Quantization status.
     */
    public enum Status {
        PENDING, RUNNING, COMPLETED, FAILED
    }

    /**
     * Builder for QuantStats.
     */
    public static class Builder {
        private long originalSizeBytes = 0L;
        private long quantizedSizeBytes = 0L;
        private long peakMemoryBytes = 0L;
        private int tensorCount = 0;
        private long paramCount = 0L;
        private double avgQuantErrorMse = 0.0;
        private double maxQuantError = 0.0;
        private Instant startTime = Instant.now();
        private Instant endTime;
        private Status status = Status.PENDING;
        private String errorMessage;

        public Builder originalSizeBytes(long originalSizeBytes) {
            this.originalSizeBytes = originalSizeBytes;
            return this;
        }

        public Builder quantizedSizeBytes(long quantizedSizeBytes) {
            this.quantizedSizeBytes = quantizedSizeBytes;
            return this;
        }

        public Builder peakMemoryBytes(long peakMemoryBytes) {
            this.peakMemoryBytes = peakMemoryBytes;
            return this;
        }

        public Builder tensorCount(int tensorCount) {
            this.tensorCount = tensorCount;
            return this;
        }

        public Builder paramCount(long paramCount) {
            this.paramCount = paramCount;
            return this;
        }

        public Builder avgQuantErrorMse(double avgQuantErrorMse) {
            this.avgQuantErrorMse = avgQuantErrorMse;
            return this;
        }

        public Builder maxQuantError(double maxQuantError) {
            this.maxQuantError = maxQuantError;
            return this;
        }

        public Builder startTime(Instant startTime) {
            this.startTime = startTime;
            return this;
        }

        public Builder endTime(Instant endTime) {
            this.endTime = endTime;
            return this;
        }

        public Builder status(Status status) {
            this.status = status;
            return this;
        }

        public Builder errorMessage(String errorMessage) {
            this.errorMessage = errorMessage;
            return this;
        }

        public QuantStats build() {
            return new QuantStats(this);
        }
    }
}
