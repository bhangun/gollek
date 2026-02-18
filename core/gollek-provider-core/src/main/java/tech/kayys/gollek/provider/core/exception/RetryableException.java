package tech.kayys.gollek.provider.core.exception;

import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.spi.error.ErrorCode;

/**
 * Exception that indicates the operation can be retried.
 *
 * @author Bhangun
 * @since 1.0.0
 */
public class RetryableException extends InferenceException {

    private final int maxRetries;
    private final long backoffMs;

    /**
     * Creates a new RetryableException with default retry parameters.
     *
     * @param errorCode The error code associated with the exception
     * @param message   The error message
     */
    public RetryableException(ErrorCode errorCode, String message) {
        this(errorCode, message, 3, 100);
    }

    /**
     * Creates a new RetryableException with custom retry parameters.
     *
     * @param errorCode  The error code associated with the exception
     * @param message    The error message
     * @param maxRetries Maximum number of retries allowed
     * @param backoffMs  Backoff time in milliseconds between retries
     */
    public RetryableException(ErrorCode errorCode, String message, int maxRetries, long backoffMs) {
        super(errorCode, message);

        // Validate parameters
        if (maxRetries < 0) {
            throw new IllegalArgumentException("Max retries must be non-negative");
        }
        if (backoffMs < 0) {
            throw new IllegalArgumentException("Backoff time must be non-negative");
        }

        this.maxRetries = maxRetries;
        this.backoffMs = backoffMs;
        addContext("maxRetries", maxRetries);
        addContext("backoffMs", backoffMs);
    }

    /**
     * Creates a new RetryableException with a cause and default retry parameters.
     *
     * @param errorCode The error code associated with the exception
     * @param message   The error message
     * @param cause     The underlying cause of the exception
     */
    public RetryableException(ErrorCode errorCode, String message, Throwable cause) {
        this(errorCode, message, cause, 3, 100);
    }

    /**
     * Creates a new RetryableException with a cause and custom retry parameters.
     *
     * @param errorCode  The error code associated with the exception
     * @param message    The error message
     * @param cause      The underlying cause of the exception
     * @param maxRetries Maximum number of retries allowed
     * @param backoffMs  Backoff time in milliseconds between retries
     */
    public RetryableException(ErrorCode errorCode, String message, Throwable cause, int maxRetries, long backoffMs) {
        super(errorCode, message, cause);

        // Validate parameters
        if (maxRetries < 0) {
            throw new IllegalArgumentException("Max retries must be non-negative");
        }
        if (backoffMs < 0) {
            throw new IllegalArgumentException("Backoff time must be non-negative");
        }

        this.maxRetries = maxRetries;
        this.backoffMs = backoffMs;
        addContext("maxRetries", maxRetries);
        addContext("backoffMs", backoffMs);
    }

    /**
     * Gets the maximum number of retries allowed.
     *
     * @return The maximum number of retries
     */
    public int getMaxRetries() {
        return maxRetries;
    }

    /**
     * Gets the backoff time in milliseconds between retries.
     *
     * @return The backoff time in milliseconds
     */
    public long getBackoffMs() {
        return backoffMs;
    }
}
