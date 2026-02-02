package tech.kayys.golek.client.exception;

/**
 * Exception for rate limiting errors.
 */
public class RateLimitException extends GolekClientException {
    
    private final int retryAfterSeconds;
    
    public RateLimitException(String message, int retryAfterSeconds) {
        super("RATE_LIMIT_ERROR", message);
        this.retryAfterSeconds = retryAfterSeconds;
    }
    
    public RateLimitException(String message, int retryAfterSeconds, Throwable cause) {
        super("RATE_LIMIT_ERROR", message, cause);
        this.retryAfterSeconds = retryAfterSeconds;
    }
    
    public int getRetryAfterSeconds() {
        return retryAfterSeconds;
    }
}