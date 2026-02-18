package tech.kayys.gollek.sdk.core.exception;

import java.time.Duration;

/**
 * Exception indicating quota/rate limit exceeded.
 */
public class QuotaExceededException extends RetryableException {
    
    private final Duration retryAfter;
    
    public QuotaExceededException(String message, Duration retryAfter) {
        super("QUOTA_EXCEEDED", message);
        this.retryAfter = retryAfter;
    }
    
    public Duration getRetryAfter() {
        return retryAfter;
    }
}
