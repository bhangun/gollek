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

package tech.kayys.gollek.plugin.optimization;

import java.time.Instant;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;

/**
 * Optimization health status with detailed information.
 *
 * @since 2.0.0
 */
public final class OptimizationHealth {

    private final boolean healthy;
    private final String message;
    private final Map<String, Object> details;
    private final Instant timestamp;

    private OptimizationHealth(boolean healthy, String message, Map<String, Object> details, Instant timestamp) {
        this.healthy = healthy;
        this.message = message;
        this.details = Collections.unmodifiableMap(details);
        this.timestamp = timestamp != null ? timestamp : Instant.now();
    }

    /**
     * Check if optimization is healthy.
     */
    public boolean isHealthy() {
        return healthy;
    }

    /**
     * Get health message.
     */
    public String getMessage() {
        return message;
    }

    /**
     * Get health details.
     */
    public Map<String, Object> getDetails() {
        return details;
    }

    /**
     * Get timestamp.
     */
    public Instant getTimestamp() {
        return timestamp;
    }

    /**
     * Create healthy status.
     */
    public static OptimizationHealth healthy() {
        return new OptimizationHealth(true, "OK", Map.of(), Instant.now());
    }

    /**
     * Create healthy status with details.
     */
    public static OptimizationHealth healthy(Map<String, Object> details) {
        return new OptimizationHealth(true, "OK", details, Instant.now());
    }

    /**
     * Create unhealthy status.
     */
    public static OptimizationHealth unhealthy(String message) {
        return new OptimizationHealth(false, Objects.requireNonNull(message), Map.of(), Instant.now());
    }

    /**
     * Create unhealthy status with details.
     */
    public static OptimizationHealth unhealthy(String message, Map<String, Object> details) {
        return new OptimizationHealth(false, message, details, Instant.now());
    }

    /**
     * Create builder.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for OptimizationHealth.
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
            ((java.util.HashMap<String, Object>)this.details).put(key, value);
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public OptimizationHealth build() {
            return new OptimizationHealth(healthy, message, details, timestamp);
        }
    }
}
