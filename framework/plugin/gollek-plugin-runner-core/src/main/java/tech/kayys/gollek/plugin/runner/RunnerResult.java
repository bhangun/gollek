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
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Execution result from runner operation with status, data, and metadata.
 *
 * @param <T> Result data type
 * @since 2.0.0
 */
public final class RunnerResult<T> {

    private final Status status;
    private final T data;
    private final String errorMessage;
    private final Map<String, Object> metadata;
    private final String executionId;
    private final Instant startTime;
    private final Instant endTime;

    private RunnerResult(Builder<T> builder) {
        this.status = builder.status;
        this.data = builder.data;
        this.errorMessage = builder.errorMessage;
        this.metadata = Map.copyOf(builder.metadata);
        this.executionId = builder.executionId;
        this.startTime = builder.startTime;
        this.endTime = builder.endTime;
    }

    /**
     * Get execution status.
     */
    public Status getStatus() {
        return status;
    }

    /**
     * Get result data.
     */
    public T getData() {
        return data;
    }

    /**
     * Get error message.
     */
    public Optional<String> getErrorMessage() {
        return Optional.ofNullable(errorMessage);
    }

    /**
     * Get result metadata.
     */
    public Map<String, Object> getMetadata() {
        return metadata;
    }

    /**
     * Get execution ID.
     */
    public String getExecutionId() {
        return executionId;
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
     * Get execution duration.
     */
    public Duration getDuration() {
        if (startTime != null && endTime != null) {
            return Duration.between(startTime, endTime);
        }
        return Duration.ZERO;
    }

    /**
     * Check if successful.
     */
    public boolean isSuccess() {
        return status == Status.SUCCESS;
    }

    /**
     * Check if failed.
     */
    public boolean isFailed() {
        return status == Status.FAILED;
    }

    /**
     * Get typed data.
     */
    @SuppressWarnings("unchecked")
    public <R> Optional<R> getDataAs(Class<R> type) {
        if (data == null) {
            return Optional.empty();
        }
        if (type.isInstance(data)) {
            return Optional.of((R) data);
        }
        throw new ClassCastException(
                "Cannot cast " + data.getClass().getName() + " to " + type.getName());
    }

    /**
     * Create success result.
     */
    public static <T> RunnerResult<T> success(T data) {
        return new Builder<T>().status(Status.SUCCESS).data(data).build();
    }

    /**
     * Create success result with metadata.
     */
    public static <T> RunnerResult<T> success(T data, Map<String, Object> metadata) {
        return new Builder<T>().status(Status.SUCCESS).data(data).metadata(metadata).build();
    }

    /**
     * Create failed result.
     */
    public static <T> RunnerResult<T> failed(String errorMessage) {
        return new Builder<T>().status(Status.FAILED).errorMessage(errorMessage).build();
    }

    /**
     * Create failed result with exception.
     */
    public static <T> RunnerResult<T> failed(Exception exception) {
        return new Builder<T>()
                .status(Status.FAILED)
                .errorMessage(exception.getMessage())
                .metadata(Map.of(
                        "exception", exception.getClass().getName(),
                        "stacktrace", getStackTrace(exception)))
                .build();
    }

    private static String getStackTrace(Exception e) {
        StringBuilder sb = new StringBuilder();
        for (StackTraceElement element : e.getStackTrace()) {
            sb.append(element.toString()).append("\n");
        }
        return sb.toString();
    }

    /**
     * Create builder.
     */
    public static <T> Builder<T> builder() {
        return new Builder<>();
    }

    /**
     * Execution status.
     */
    public enum Status {
        SUCCESS,
        FAILED,
        CANCELLED,
        TIMEOUT
    }

    /**
     * Builder for RunnerResult.
     */
    public static class Builder<T> {
        private Status status = Status.SUCCESS;
        private T data;
        private String errorMessage;
        private Map<String, Object> metadata = new ConcurrentHashMap<>();
        private String executionId = UUID.randomUUID().toString();
        private Instant startTime = Instant.now();
        private Instant endTime;

        public Builder<T> status(Status status) {
            this.status = status;
            return this;
        }

        public Builder<T> data(T data) {
            this.data = data;
            return this;
        }

        public Builder<T> errorMessage(String errorMessage) {
            this.errorMessage = errorMessage;
            return this;
        }

        public Builder<T> metadata(Map<String, Object> metadata) {
            this.metadata.putAll(metadata);
            return this;
        }

        public Builder<T> metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public Builder<T> executionId(String executionId) {
            this.executionId = executionId;
            return this;
        }

        public Builder<T> startTime(Instant startTime) {
            this.startTime = startTime;
            return this;
        }

        public Builder<T> endTime(Instant endTime) {
            this.endTime = endTime;
            return this;
        }

        public RunnerResult<T> build() {
            if (status == Status.SUCCESS && data == null) {
                throw new IllegalStateException("Success result must have data");
            }
            if (status == Status.FAILED && errorMessage == null) {
                throw new IllegalStateException("Failed result must have error message");
            }
            if (endTime == null) {
                endTime = Instant.now();
            }
            return new RunnerResult<>(this);
        }
    }
}
