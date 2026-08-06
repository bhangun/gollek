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

import java.time.Instant;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;

/**
 * Kernel health status with detailed information.
 *
 * @since 2.0.0
 */
public final class KernelHealth {

    private final boolean healthy;
    private final String message;
    private final Map<String, Object> details;
    private final Instant timestamp;

    private KernelHealth(boolean healthy, String message, Map<String, Object> details, Instant timestamp) {
        this.healthy = healthy;
        this.message = message;
        this.details = Collections.unmodifiableMap(details);
        this.timestamp = timestamp != null ? timestamp : Instant.now();
    }

    /**
     * Check if kernel is healthy.
     *
     * @return true if healthy
     */
    public boolean isHealthy() {
        return healthy;
    }

    /**
     * Get health message.
     *
     * @return health message
     */
    public String getMessage() {
        return message;
    }

    /**
     * Get health details.
     *
     * @return immutable details map
     */
    public Map<String, Object> getDetails() {
        return details;
    }

    /**
     * Get timestamp.
     *
     * @return health check timestamp
     */
    public Instant getTimestamp() {
        return timestamp;
    }

    /**
     * Create healthy status.
     *
     * @return healthy status
     */
    public static KernelHealth healthy() {
        return new KernelHealth(true, "OK", Map.of(), Instant.now());
    }

    /**
     * Create healthy status with details.
     *
     * @param details health details
     * @return healthy status
     */
    public static KernelHealth healthy(Map<String, Object> details) {
        return new KernelHealth(true, "OK", details, Instant.now());
    }

    /**
     * Create unhealthy status.
     *
     * @param message error message
     * @return unhealthy status
     */
    public static KernelHealth unhealthy(String message) {
        return new KernelHealth(false, Objects.requireNonNull(message), Map.of(), Instant.now());
    }

    /**
     * Create unhealthy status with details.
     *
     * @param message error message
     * @param details error details
     * @return unhealthy status
     */
    public static KernelHealth unhealthy(String message, Map<String, Object> details) {
        return new KernelHealth(false, message, details, Instant.now());
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
     * Builder for KernelHealth.
     */
    public static class Builder {
        private boolean healthy = true;
        private String message = "OK";
        private Map<String, Object> details = Map.of();
        private Instant timestamp = Instant.now();

        public Builder healthy(boolean healthy) {
            this.healthy = healthy;
            return this;
        }

        public Builder message(String message) {
            this.message = message;
            return this;
        }

        public Builder details(Map<String, Object> details) {
            this.details = Map.copyOf(details);
            return this;
        }

        public Builder detail(String key, Object value) {
            if (this.details.isEmpty()) {
                this.details = new java.util.HashMap<>();
            }
            ((Map<String, Object>)this.details).put(key, value);
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public KernelHealth build() {
            return new KernelHealth(healthy, message, details, timestamp);
        }
    }
}
