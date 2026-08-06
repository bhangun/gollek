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

package tech.kayys.gollek.plugin.kernel;

import java.util.Objects;

/**
 * Kernel configuration with device settings, memory management, and
 * execution options.
 *
 * @param deviceId GPU device ID (0-based)
 * @param memoryFraction Fraction of GPU memory to allocate (0.0-1.0)
 * @param allowGrowth Allow dynamic memory growth
 * @param computeMode Compute mode (default, exclusive, prohibited)
 * @param streaming Enable streaming execution
 * @param timeoutMs Operation timeout in milliseconds
 * @since 2.0.0
 */
public record KernelConfig(
    int deviceId,
    float memoryFraction,
    boolean allowGrowth,
    String computeMode,
    boolean streaming,
    long timeoutMs
) {
    /**
     * Create default configuration.
     */
    public KernelConfig() {
        this(0, 0.9f, false, "default", false, 300000L);
    }

    /**
     * Create configuration with builder.
     */
    public KernelConfig(Builder builder) {
        this(
            builder.deviceId,
            builder.memoryFraction,
            builder.allowGrowth,
            builder.computeMode,
            builder.streaming,
            builder.timeoutMs
        );
    }

    /**
     * Validate configuration.
     *
     * @return true if configuration is valid
     */
    public boolean isValid() {
        return deviceId >= 0 &&
               memoryFraction > 0.0f && memoryFraction <= 1.0f &&
               computeMode != null && !computeMode.isBlank() &&
               timeoutMs > 0;
    }

    /**
     * Get default configuration.
     *
     * @return default configuration
     */
    public static KernelConfig defaultConfig() {
        return new KernelConfig();
    }

    /**
     * Create builder.
     *
     * @return new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for KernelConfig.
     */
    public static class Builder {
        private int deviceId = 0;
        private float memoryFraction = 0.9f;
        private boolean allowGrowth = false;
        private String computeMode = "default";
        private boolean streaming = false;
        private long timeoutMs = 300000L; // 5 minutes

        public Builder deviceId(int deviceId) {
            if (deviceId < 0) {
                throw new IllegalArgumentException("Device ID must be non-negative");
            }
            this.deviceId = deviceId;
            return this;
        }

        public Builder memoryFraction(float memoryFraction) {
            if (memoryFraction <= 0.0f || memoryFraction > 1.0f) {
                throw new IllegalArgumentException(
                    "Memory fraction must be between 0.0 and 1.0");
            }
            this.memoryFraction = memoryFraction;
            return this;
        }

        public Builder allowGrowth(boolean allowGrowth) {
            this.allowGrowth = allowGrowth;
            return this;
        }

        public Builder computeMode(String computeMode) {
            Objects.requireNonNull(computeMode, "Compute mode cannot be null");
            this.computeMode = computeMode;
            return this;
        }

        public Builder streaming(boolean streaming) {
            this.streaming = streaming;
            return this;
        }

        public Builder timeoutMs(long timeoutMs) {
            if (timeoutMs <= 0) {
                throw new IllegalArgumentException("Timeout must be positive");
            }
            this.timeoutMs = timeoutMs;
            return this;
        }

        public KernelConfig build() {
            return new KernelConfig(this);
        }
    }
}
