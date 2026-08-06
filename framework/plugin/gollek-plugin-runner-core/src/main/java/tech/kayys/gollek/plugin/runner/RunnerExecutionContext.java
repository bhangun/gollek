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

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Execution context for runner operations with lifecycle management.
 *
 * @since 2.0.0
 */
public final class RunnerExecutionContext {

    private final String executionId;
    private final String operationName;
    private final Instant startTime;
    private final RunnerConfig config;
    private final Map<String, Object> resources;
    private final AtomicBoolean cancelled;
    private final AtomicBoolean completed;
    private volatile Instant endTime;
    private volatile String cancellationReason;

    private RunnerExecutionContext(Builder builder) {
        this.executionId = builder.executionId;
        this.operationName = builder.operationName;
        this.startTime = builder.startTime != null ? builder.startTime : Instant.now();
        this.config = builder.config;
        this.resources = new ConcurrentHashMap<>(builder.resources);
        this.cancelled = new AtomicBoolean(false);
        this.completed = new AtomicBoolean(false);
    }

    /**
     * Get execution ID.
     */
    public String getExecutionId() {
        return executionId;
    }

    /**
     * Get operation name.
     */
    public String getOperationName() {
        return operationName;
    }

    /**
     * Get start time.
     */
    public Instant getStartTime() {
        return startTime;
    }

    /**
     * Get end time.
     */
    public Instant getEndTime() {
        return endTime;
    }

    /**
     * Get duration.
     */
    public Duration getDuration() {
        Instant end = endTime != null ? endTime : Instant.now();
        return Duration.between(startTime, end);
    }

    /**
     * Get config.
     */
    public RunnerConfig getConfig() {
        return config;
    }

    /**
     * Get resource.
     */
    public Optional<Object> getResource(String key) {
        return Optional.ofNullable(resources.get(key));
    }

    /**
     * Get typed resource.
     */
    @SuppressWarnings("unchecked")
    public <T> Optional<T> getResource(String key, Class<T> type) {
        Object resource = resources.get(key);
        if (resource == null) {
            return Optional.empty();
        }
        if (type.isInstance(resource)) {
            return Optional.of((T) resource);
        }
        throw new ClassCastException("Resource is not of type " + type.getSimpleName());
    }

    /**
     * Get all resources.
     */
    public Map<String, Object> getResources() {
        return Map.copyOf(resources);
    }

    /**
     * Check if cancelled.
     */
    public boolean isCancelled() {
        return cancelled.get();
    }

    /**
     * Check if completed.
     */
    public boolean isCompleted() {
        return completed.get();
    }

    /**
     * Get cancellation reason.
     */
    public Optional<String> getCancellationReason() {
        return Optional.ofNullable(cancellationReason);
    }

    /**
     * Mark completed.
     */
    public void markCompleted() {
        if (cancelled.get()) {
            throw new IllegalStateException("Cannot complete cancelled execution");
        }
        if (completed.compareAndSet(false, true)) {
            this.endTime = Instant.now();
        }
    }

    /**
     * Cancel execution.
     */
    public boolean cancel(String reason) {
        if (completed.get()) {
            return false;
        }
        if (cancelled.compareAndSet(false, true)) {
            this.cancellationReason = reason;
            this.endTime = Instant.now();
            return true;
        }
        return false;
    }

    /**
     * Create empty context.
     */
    public static RunnerExecutionContext empty() {
        return builder().build();
    }

    /**
     * Create builder.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for RunnerExecutionContext.
     */
    public static class Builder {
        private String executionId = UUID.randomUUID().toString();
        private String operationName = "unknown";
        private Instant startTime;
        private RunnerConfig config = RunnerConfig.defaultConfig();
        private Map<String, Object> resources = Map.of();

        public Builder executionId(String executionId) {
            this.executionId = Objects.requireNonNull(executionId);
            return this;
        }

        public Builder operationName(String operationName) {
            this.operationName = Objects.requireNonNull(operationName);
            return this;
        }

        public Builder startTime(Instant startTime) {
            this.startTime = startTime;
            return this;
        }

        public Builder config(RunnerConfig config) {
            this.config = config;
            return this;
        }

        public Builder resources(Map<String, Object> resources) {
            this.resources = new ConcurrentHashMap<>(resources);
            return this;
        }

        public Builder resource(String key, Object value) {
            if (this.resources.isEmpty()) {
                this.resources = new ConcurrentHashMap<>();
            }
            ((ConcurrentHashMap<String, Object>)this.resources).put(key, value);
            return this;
        }

        public RunnerExecutionContext build() {
            return new RunnerExecutionContext(this);
        }
    }
}
