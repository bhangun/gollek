package tech.kayys.golek.sdk.core.exception;

/**
 * Exception indicating a transient failure that can be retried.
 */
public class RetryableException extends SdkException {
    
    public RetryableException(String message) {
        super("RETRYABLE_ERROR", message);
    }
    
    public RetryableException(String message, Throwable cause) {
        super("RETRYABLE_ERROR", message, cause);
    }
    
    public RetryableException(String errorCode, String message) {
        super(errorCode, message);
    }
    
    public RetryableException(String errorCode, String message, Throwable cause) {
        super(errorCode, message, cause);
    }
}
