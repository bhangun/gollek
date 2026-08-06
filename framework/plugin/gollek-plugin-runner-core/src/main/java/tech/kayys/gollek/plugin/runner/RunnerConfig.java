/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package tech.kayys.gollek.plugin.runner;

/**
 * Runner configuration with model loading and execution settings.
 *
 * @param contextSize    Context size (tokens)
 * @param threads        Number of CPU threads
 * @param nGpuLayers     Number of layers to offload to GPU (-1 for all)
 * @param batchSize      Batch size for inference
 * @param threadsBatch   Number of threads for batch processing
 * @param flashAttention Enable flash attention
 * @param memoryLimit    Memory limit in MB
 * @since 2.0.0
 */
public record RunnerConfig(
        int contextSize,
        int threads,
        int nGpuLayers,
        int batchSize,
        int threadsBatch,
        boolean flashAttention,
        long memoryLimit) {
    /**
     * Create default configuration.
     */
    public RunnerConfig() {
        this(4096, 4, -1, 1, 1, false, 0L);
    }

    /**
     * Create configuration with builder.
     */
    public RunnerConfig(Builder builder) {
        this(
                builder.contextSize,
                builder.threads,
                builder.nGpuLayers,
                builder.batchSize,
                builder.threadsBatch,
                builder.flashAttention,
                builder.memoryLimit);
    }

    /**
     * Validate configuration.
     */
    public boolean isValid() {
        return contextSize > 0 &&
                threads > 0 &&
                batchSize > 0 &&
                memoryLimit >= 0;
    }

    /**
     * Get default configuration.
     */
    public static RunnerConfig defaultConfig() {
        return new RunnerConfig();
    }

    /**
     * Create builder.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for RunnerConfig.
     */
    public static class Builder {
        private int contextSize = 4096;
        private int threads = 4;
        private int nGpuLayers = -1;
        private int batchSize = 1;
        private int threadsBatch = 1;
        private boolean flashAttention = false;
        private long memoryLimit = 0L;

        public Builder contextSize(int contextSize) {
            if (contextSize <= 0) {
                throw new IllegalArgumentException("Context size must be positive");
            }
            this.contextSize = contextSize;
            return this;
        }

        public Builder threads(int threads) {
            if (threads <= 0) {
                throw new IllegalArgumentException("Threads must be positive");
            }
            this.threads = threads;
            return this;
        }

        public Builder nGpuLayers(int nGpuLayers) {
            this.nGpuLayers = nGpuLayers;
            return this;
        }

        public Builder batchSize(int batchSize) {
            if (batchSize <= 0) {
                throw new IllegalArgumentException("Batch size must be positive");
            }
            this.batchSize = batchSize;
            return this;
        }

        public Builder threadsBatch(int threadsBatch) {
            if (threadsBatch <= 0) {
                throw new IllegalArgumentException("Batch threads must be positive");
            }
            this.threadsBatch = threadsBatch;
            return this;
        }

        public Builder flashAttention(boolean flashAttention) {
            this.flashAttention = flashAttention;
            return this;
        }

        public Builder memoryLimit(long memoryLimit) {
            if (memoryLimit < 0) {
                throw new IllegalArgumentException("Memory limit must be non-negative");
            }
            this.memoryLimit = memoryLimit;
            return this;
        }

        public RunnerConfig build() {
            return new RunnerConfig(this);
        }
    }
}
