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
 * Base exception for provider-related errors.
 *
 * @since 2.1.0
 */
public class ProviderException extends RuntimeException {

    private final String providerId;
    private final ErrorCode errorCode;
    private final boolean retryable;
    private final Map<String, String> context = new HashMap<>();

    // Simple constructors for backward compatibility
    public ProviderException(String message) {
        super(message);
        this.providerId = null;
        this.errorCode = null;
        this.retryable = false;
    }

    public ProviderException(String message, Throwable cause) {
        super(message, cause);
        this.providerId = null;
        this.errorCode = null;
        this.retryable = false;
    }

    // Full constructor for provider-core
    public ProviderException(String providerId, String message, Throwable cause, 
                            ErrorCode errorCode, boolean retryable) {
        super(message, cause);
        this.providerId = providerId;
        this.errorCode = errorCode;
        this.retryable = retryable;
    }

    public String getProviderId() {
        return providerId;
    }

    public ErrorCode getErrorCode() {
        return errorCode;
    }

    public boolean isRetryable() {
        return retryable;
    }

    public void addContext(String key, String value) {
        context.put(key, value);
    }

    public Map<String, String> getContext() {
        return new HashMap<>(context);
    }

    /**
     * Exception thrown when provider initialization fails.
     */
    public static class ProviderInitializationException extends ProviderException {
        public ProviderInitializationException(String message) {
            super(message);
        }

        public ProviderInitializationException(String providerId, String message) {
            super(providerId, message, null, ErrorCode.INIT_RUNNER_FAILED, false);
        }

        public ProviderInitializationException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    /**
     * Exception thrown when provider authentication fails.
     */
    public static class ProviderAuthenticationException extends ProviderException {
        public ProviderAuthenticationException(String message) {
            super(message);
        }

        public ProviderAuthenticationException(String providerId, String message) {
            super(providerId, message, null, ErrorCode.PROVIDER_AUTH_FAILED, false);
        }

        public ProviderAuthenticationException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    /**
     * Exception thrown when provider is unavailable.
     */
    public static class ProviderUnavailableException extends ProviderException {
        public ProviderUnavailableException(String message) {
            super(message);
        }

        public ProviderUnavailableException(String providerId, String message) {
            super(providerId, message, null, ErrorCode.PROVIDER_UNAVAILABLE, true);
        }

        public ProviderUnavailableException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
