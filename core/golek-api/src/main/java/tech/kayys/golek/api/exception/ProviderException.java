package tech.kayys.golek.api.exception;

import tech.kayys.golek.api.ErrorPayload;

/**
 * Base exception for all provider errors.
 */
public class ProviderException extends RuntimeException {

    private final String providerId;
    private final boolean retryable;
    private final ErrorPayload errorPayload;
    private final String errorCode;

    public ProviderException(String message) {
        this(null, message, null, null, false);
    }

    public ProviderException(String message, Throwable cause) {
        this(null, message, cause, null, false);
    }

    public ProviderException(String providerId, String message) {
        this(providerId, message, null, null, false);
    }

    public ProviderException(String providerId, String message, Throwable cause) {
        this(providerId, message, cause, null, false);
    }

    public ProviderException(String providerId, String message, Throwable cause, boolean retryable) {
        this(providerId, message, cause, null, retryable);
    }

    public ProviderException(
            String providerId,
            String message,
            Throwable cause,
            String errorCode,
            boolean retryable) {
        super(message, cause);
        this.providerId = providerId;
        this.retryable = retryable;
        this.errorCode = errorCode;
        this.errorPayload = ErrorPayload.builder()
                .type("ProviderError")
                .message(message)
                .retryable(retryable)
                .originNode(providerId)
                .detail("providerId", providerId)
                .build();
    }

    public String getProviderId() {
        return providerId;
    }

    public ErrorPayload getErrorPayload() {
        return errorPayload;
    }

    public boolean isRetryable() {
        return retryable;
    }

    public String getErrorCode() {
        return errorCode;
    }

    // Specific exception types
    public static class ProviderInitializationException extends ProviderException {
        public ProviderInitializationException(String message, Throwable cause) {
            super(null, message, cause, null, false);
        }

        public ProviderInitializationException(String providerId, String message, Throwable cause) {
            super(providerId, message, cause, null, false);
        }
    }

    public static class ProviderUnavailableException extends ProviderException {
        public ProviderUnavailableException(String providerId, String message) {
            super(providerId, message, null, "PROVIDER_UNAVAILABLE", true);
        }
    }

    public static class ProviderTimeoutException extends ProviderException {
        public ProviderTimeoutException(String providerId, String message) {
            super(providerId, message, null, "PROVIDER_TIMEOUT", true);
        }
    }

    public static class ProviderQuotaExceededException extends ProviderException {
        public ProviderQuotaExceededException(String providerId, String message) {
            super(providerId, message, null, "QUOTA_EXCEEDED", false);
        }
    }

    public static class ProviderAuthenticationException extends ProviderException {
        public ProviderAuthenticationException(String providerId, String message) {
            super(providerId, message, null, "AUTH_FAILED", false);
        }
    }

    public static class ProviderRateLimitException extends ProviderException {
        private final long retryAfterMs;

        public ProviderRateLimitException(String providerId, String message, long retryAfterMs) {
            super(providerId, message, null, "RATE_LIMIT_EXCEEDED", true);
            this.retryAfterMs = retryAfterMs;
        }

        public long getRetryAfterMs() {
            return retryAfterMs;
        }
    }
}
