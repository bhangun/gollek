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
 */

package tech.kayys.gollek.spi.exception;

import tech.kayys.gollek.error.ErrorCode;
import java.util.HashMap;
import java.util.Map;

/**
 * Base exception for inference-related errors.
 *
 * @since 2.1.0
 */
public class InferenceException extends RuntimeException {

    private final ErrorCode errorCode;
    private final Map<String, String> context = new HashMap<>();

    public InferenceException(ErrorCode errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }

    public InferenceException(ErrorCode errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }

    // Simple constructors for provider-core compatibility
    public InferenceException(String message) {
        super(message);
        this.errorCode = null;
    }

    public InferenceException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = null;
    }

    public ErrorCode getErrorCode() {
        return errorCode;
    }

    /** Fluent context setter — returns {@code this} for chaining. */
    public InferenceException addContext(String key, String value) {
        context.put(key, value);
        return this;
    }

    /** Convenience overload for numeric context values. */
    public InferenceException addContext(String key, int value) {
        context.put(key, String.valueOf(value));
        return this;
    }

    /** Convenience overload for Object context values. */
    public InferenceException addContext(String key, Object value) {
        context.put(key, value != null ? value.toString() : "null");
        return this;
    }

    public Map<String, String> getContext() {
        return new HashMap<>(context);
    }
}
