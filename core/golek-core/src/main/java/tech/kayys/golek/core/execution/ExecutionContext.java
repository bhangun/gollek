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

import tech.kayys.golek.spi.context.RequestContext;
import tech.kayys.golek.spi.context.EngineContext;
import tech.kayys.golek.spi.execution.ExecutionStatus;
import tech.kayys.golek.spi.inference.InferencePhase;

import java.util.Map;
import java.util.Optional;

/**
 * Mutable execution context for a single inference request.
 * Provides access to execution token, variables, and engine context.
 */
public interface ExecutionContext {

    /**
     * Get engine context (global state)
     */
    EngineContext engine();

    /**
     * Get current execution token (immutable snapshot)
     */
    ExecutionToken token();

    /**
     * Get tenant context
     */
    RequestContext requestContext();

    /**
     * Update execution status (creates new token)
     */
    void updateStatus(ExecutionStatus status);

    /**
     * Update current phase (creates new token)
     */
    void updatePhase(InferencePhase phase);

    /**
     * Increment retry attempt
     */
    void incrementAttempt();

    /**
     * Get execution variables (mutable view)
     */
    Map<String, Object> variables();

    /**
     * Put variable
     */
    void putVariable(String key, Object value);

    /**
     * Get variable
     */
    <T> Optional<T> getVariable(String key, Class<T> type);

    /**
     * Remove variable
     */
    void removeVariable(String key);

    /**
     * Get metadata
     */
    Map<String, Object> metadata();

    /**
     * Put metadata
     */
    void putMetadata(String key, Object value);

    /**
     * Replace entire token (for state restoration)
     */
    void replaceToken(ExecutionToken newToken);

    /**
     * Check if context has error
     */
    boolean hasError();

    /**
     * Get error if present
     */
    Optional<Throwable> getError();

    /**
     * Set error
     */
    void setError(Throwable error);

    /**
     * Clear error
     */
    void clearError();
}