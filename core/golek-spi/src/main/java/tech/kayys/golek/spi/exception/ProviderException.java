package tech.kayys.golek.spi.exception;

import tech.kayys.golek.spi.error.ErrorCode;
import tech.kayys.golek.spi.error.ErrorPayload;

/**
 * Base exception for all provider errors.
 */
public class ProviderException extends RuntimeException {

    private final String providerId;
    private final boolean retryable;
    private final ErrorPayload errorPayload;
    private final String errorCode;
    private final ErrorCode errorCodeEnum;

    public ProviderException(String message) {
        this(null, message, null, (String) null, false);
    }

    public ProviderException(String message, Throwable cause) {
        this(null, message, cause, (String) null, false);
    }

    public ProviderException(String providerId, String message) {
        this(providerId, message, null, (String) null, false);
    }

    public ProviderException(String providerId, String message, Throwable cause) {
        this(providerId, message, cause, (String) null, false);
    }

    public ProviderException(String providerId, String message, Throwable cause, boolean retryable) {
        this(providerId, message, cause, (String) null, retryable);
    }

    public ProviderException(String providerId, String message, Throwable cause, ErrorCode errorCode,
            boolean retryable) {
        this(providerId, message, cause, errorCode != null ? errorCode.getCode() : null, retryable);
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
        this.errorCodeEnum = ErrorCode.fromCode(errorCode, ErrorCode.INTERNAL_ERROR);
        this.errorCode = errorCode != null ? errorCode : this.errorCodeEnum.getCode();
        this.errorPayload = ErrorPayload.builder()
                .type("ProviderError")
                .message(message)
                .retryable(retryable)
                .originNode(providerId)
                .detail("errorCode", this.errorCodeEnum.getCode())
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

    public ErrorCode getErrorCodeEnum() {
        return errorCodeEnum;
    }

    // Specific exception types
    public static class ProviderInitializationException extends ProviderException {
        public ProviderInitializationException(String message) {
            super(null, message, null, ErrorCode.PROVIDER_INIT_FAILED, false);
        }

        public ProviderInitializationException(String message, Throwable cause) {
            super(null, message, cause, ErrorCode.PROVIDER_INIT_FAILED, false);
        }

        public ProviderInitializationException(String providerId, String message) {
            super(providerId, message, null, ErrorCode.PROVIDER_INIT_FAILED, false);
        }

        public ProviderInitializationException(String providerId, String message, Throwable cause) {
            super(providerId, message, cause, ErrorCode.PROVIDER_INIT_FAILED, false);
        }
    }

    public static class ProviderUnavailableException extends ProviderException {
        public ProviderUnavailableException(String providerId, String message) {
            super(providerId, message, null, ErrorCode.PROVIDER_UNAVAILABLE, true);
        }
    }

    public static class ProviderTimeoutException extends ProviderException {
        public ProviderTimeoutException(String providerId, String message) {
            super(providerId, message, null, ErrorCode.PROVIDER_TIMEOUT, true);
        }
    }

    public static class ProviderQuotaExceededException extends ProviderException {
        public ProviderQuotaExceededException(String providerId, String message) {
            super(providerId, message, null, ErrorCode.PROVIDER_QUOTA_EXCEEDED, false);
        }
    }

    public static class ProviderAuthenticationException extends ProviderException {
        public ProviderAuthenticationException(String providerId, String message) {
            super(providerId, message, null, ErrorCode.PROVIDER_AUTH_FAILED, false);
        }
    }

    public static class ProviderRateLimitException extends ProviderException {
        private final long retryAfterMs;

        public ProviderRateLimitException(String providerId, String message, long retryAfterMs) {
            super(providerId, message, null, ErrorCode.PROVIDER_RATE_LIMITED, true);
            this.retryAfterMs = retryAfterMs;
        }

        public long getRetryAfterMs() {
            return retryAfterMs;
        }
    }
}
