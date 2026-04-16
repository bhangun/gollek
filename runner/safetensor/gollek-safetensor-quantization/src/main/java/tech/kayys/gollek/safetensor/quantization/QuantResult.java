/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QuantResult.java
 * ───────────────────────
 * Quantization result.
 */
package tech.kayys.gollek.safetensor.quantization;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Result of a quantization operation.
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class QuantResult {

    /**
     * Path to the quantized model.
     */
    private final Path quantizedModelPath;

    /**
     * Quantization statistics.
     */
    private final QuantStats stats;

    /**
     * Quantization configuration used.
     */
    private final QuantConfig config;

    /**
     * Whether quantization was successful.
     */
    private final boolean success;

    /**
     * Error message if failed.
     */
    private final String errorMessage;

    private QuantResult(Builder builder) {
        this.quantizedModelPath = builder.quantizedModelPath;
        this.stats = builder.stats;
        this.config = builder.config;
        this.success = builder.success;
        this.errorMessage = builder.errorMessage;
    }

    /**
     * Create builder for QuantResult.
     *
     * @return new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create successful result.
     *
     * @param quantizedModelPath path to quantized model
     * @param stats              quantization statistics
     * @param config             configuration used
     * @return successful result
     */
    public static QuantResult success(Path quantizedModelPath, QuantStats stats, QuantConfig config) {
        return builder()
                .quantizedModelPath(quantizedModelPath)
                .stats(stats)
                .config(config)
                .success(true)
                .build();
    }

    /**
     * Create failed result.
     *
     * @param errorMessage error message
     * @param config       configuration used
     * @return failed result
     */
    public static QuantResult failure(String errorMessage, QuantConfig config) {
        return builder()
                .config(config)
                .success(false)
                .errorMessage(errorMessage)
                .build();
    }

    public Path getQuantizedModelPath() {
        return quantizedModelPath;
    }

    public QuantStats getStats() {
        return stats;
    }

    public QuantConfig getConfig() {
        return config;
    }

    public boolean isSuccess() {
        return success;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    /**
     * Builder for QuantResult.
     */
    public static class Builder {
        private Path quantizedModelPath;
        private QuantStats stats;
        private QuantConfig config;
        private boolean success;
        private String errorMessage;

        public Builder quantizedModelPath(Path quantizedModelPath) {
            this.quantizedModelPath = quantizedModelPath;
            return this;
        }

        public Builder stats(QuantStats stats) {
            this.stats = stats;
            return this;
        }

        public Builder config(QuantConfig config) {
            this.config = config;
            return this;
        }

        public Builder success(boolean success) {
            this.success = success;
            return this;
        }

        public Builder errorMessage(String errorMessage) {
            this.errorMessage = errorMessage;
            return this;
        }

        public QuantResult build() {
            return new QuantResult(this);
        }
    }
}
