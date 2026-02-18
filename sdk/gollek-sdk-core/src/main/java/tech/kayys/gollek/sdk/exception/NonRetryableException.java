package tech.kayys.gollek.sdk.core.exception;

/**
 * Exception indicating a permanent failure that should not be retried.
 */
public class NonRetryableException extends SdkException {
    
    public NonRetryableException(String message) {
        super("NON_RETRYABLE_ERROR", message);
    }
    
    public NonRetryableException(String message, Throwable cause) {
        super("NON_RETRYABLE_ERROR", message, cause);
    }
    
    public NonRetryableException(String errorCode, String message) {
        super(errorCode, message);
    }
    
    public NonRetryableException(String errorCode, String message, Throwable cause) {
        super(errorCode, message, cause);
    }
}
