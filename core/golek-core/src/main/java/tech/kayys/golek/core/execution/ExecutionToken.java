/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 *
 * @author Bhangun
 */

package tech.kayys.golek.core.execution;

import java.io.Serializable;
import java.time.Instant;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

import tech.kayys.golek.spi.inference.InferencePhase;
import tech.kayys.golek.spi.execution.ExecutionStatus;
import tech.kayys.golek.core.exception.IllegalStateTransitionException;

/**
 * Immutable execution token representing the current state
 * of an inference request execution.
 * 
 * This is the single source of truth for execution state.
 * Serializable, persistable, and rehydration-safe.
 */
public final class ExecutionToken implements Serializable {

    private static final long serialVersionUID = 1L;

    private final String executionId;
    private final String requestId;
    private final ExecutionStatus status;
    private final InferencePhase currentPhase;
    private final int attempt;
    private final Instant createdAt;
    private final Instant lastUpdated;
    private final Map<String, Object> variables;
    private final Map<String, Object> metadata;

    private ExecutionToken(Builder builder) {
        this.executionId = builder.executionId;
        this.requestId = builder.requestId;
        this.status = builder.status;
        this.currentPhase = builder.currentPhase;
        this.attempt = builder.attempt;
        this.createdAt = builder.createdAt;
        this.lastUpdated = builder.lastUpdated;
        this.variables = new ConcurrentHashMap<>(builder.variables);
        this.metadata = new ConcurrentHashMap<>(builder.metadata);
    }

    // Getters
    public String getExecutionId() {
        return executionId;
    }

    public String getRequestId() {
        return requestId;
    }

    public ExecutionStatus getStatus() {
        return status;
    }

    public InferencePhase getCurrentPhase() {
        return currentPhase;
    }

    public int getAttempt() {
        return attempt;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getLastUpdated() {
        return lastUpdated;
    }

    public Map<String, Object> getVariables() {
        return Collections.unmodifiableMap(variables);
    }

    public Map<String, Object> getMetadata() {
        return Collections.unmodifiableMap(metadata);
    }

    /**
     * Create new token with updated status
     */
    public ExecutionToken withStatus(ExecutionStatus newStatus) {
        if (!status.canTransitionTo(newStatus)) {
            throw new IllegalStateTransitionException(
                    "Cannot transition from " + status + " to " + newStatus);
        }
        return toBuilder()
                .status(newStatus)
                .lastUpdated(Instant.now())
                .build();
    }

    /**
     * Create new token with updated phase
     */
    public ExecutionToken withPhase(InferencePhase newPhase) {
        return toBuilder()
                .currentPhase(newPhase)
                .lastUpdated(Instant.now())
                .build();
    }

    /**
     * Create new token with incremented attempt
     */
    public ExecutionToken withNextAttempt() {
        return toBuilder()
                .attempt(attempt + 1)
                .lastUpdated(Instant.now())
                .build();
    }

    /**
     * Create new token with updated variable
     */
    public ExecutionToken withVariable(String key, Object value) {
        Map<String, Object> newVars = new ConcurrentHashMap<>(variables);
        newVars.put(key, value);
        return toBuilder()
                .variables(newVars)
                .lastUpdated(Instant.now())
                .build();
    }

    /**
     * Create new token with updated metadata
     */
    public ExecutionToken withMetadata(String key, Object value) {
        Map<String, Object> newMeta = new ConcurrentHashMap<>(metadata);
        newMeta.put(key, value);
        return toBuilder()
                .metadata(newMeta)
                .lastUpdated(Instant.now())
                .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public Builder toBuilder() {
        return new Builder()
                .executionId(executionId)
                .requestId(requestId)
                .status(status)
                .currentPhase(currentPhase)
                .attempt(attempt)
                .createdAt(createdAt)
                .lastUpdated(lastUpdated)
                .variables(variables)
                .metadata(metadata);
    }

    public static class Builder {
        private String executionId;
        private String requestId;
        private ExecutionStatus status = ExecutionStatus.CREATED;
        private InferencePhase currentPhase = InferencePhase.PRE_VALIDATE;
        private int attempt = 0;
        private Instant createdAt = Instant.now();
        private Instant lastUpdated = Instant.now();
        private Map<String, Object> variables = new ConcurrentHashMap<>();
        private Map<String, Object> metadata = new ConcurrentHashMap<>();

        public Builder executionId(String executionId) {
            this.executionId = executionId;
            return this;
        }

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder status(ExecutionStatus status) {
            this.status = status;
            return this;
        }

        public Builder currentPhase(InferencePhase currentPhase) {
            this.currentPhase = currentPhase;
            return this;
        }

        public Builder attempt(int attempt) {
            this.attempt = attempt;
            return this;
        }

        public Builder createdAt(Instant createdAt) {
            this.createdAt = createdAt;
            return this;
        }

        public Builder lastUpdated(Instant lastUpdated) {
            this.lastUpdated = lastUpdated;
            return this;
        }

        public Builder variables(Map<String, Object> variables) {
            this.variables = new ConcurrentHashMap<>(variables);
            return this;
        }

        public Builder variable(String key, Object value) {
            this.variables.put(key, value);
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = new ConcurrentHashMap<>(metadata);
            return this;
        }

        public Builder metadataEntry(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public ExecutionToken build() {
            if (executionId == null) {
                executionId = UUID.randomUUID().toString();
            }
            if (requestId == null) {
                throw new IllegalStateException("requestId is required");
            }
            return new ExecutionToken(this);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof ExecutionToken that))
            return false;
        return executionId.equals(that.executionId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(executionId);
    }

    @Override
    public String toString() {
        return "ExecutionToken{" +
                "executionId='" + executionId + '\'' +
                ", status=" + status +
                ", phase=" + currentPhase +
                ", attempt=" + attempt +
                '}';
    }
}