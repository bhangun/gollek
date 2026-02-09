package tech.kayys.golek.converter;

/**
 * Exception thrown during GGUF conversion operations.
 * 
 * @author Bhangun
 * @version 1.0.0
 */
public class GGUFException extends RuntimeException {

    /**
     * Error code from native library.
     */
    private final int errorCode;

    /**
     * Create exception with message.
     * 
     * @param message error message
     */
    public GGUFException(String message) {
        super(message);
        this.errorCode = 0;
    }

    /**
     * Create exception with message and cause.
     * 
     * @param message error message
     * @param cause   underlying cause
     */
    public GGUFException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = 0;
    }

    /**
     * Create exception with message and error code.
     * 
     * @param message   error message
     * @param errorCode native error code
     */
    public GGUFException(String message, int errorCode) {
        super(message);
        this.errorCode = errorCode;
    }

    /**
     * Create exception with message, cause, and error code.
     * 
     * @param message   error message
     * @param cause     underlying cause
     * @param errorCode native error code
     */
    public GGUFException(String message, Throwable cause, int errorCode) {
        super(message, cause);
        this.errorCode = errorCode;
    }

    /**
     * Get native error code.
     * 
     * @return error code (0 if not from native code)
     */
    public int getErrorCode() {
        return errorCode;
    }

    /**
     * Check if this is a cancellation error.
     * 
     * @return true if cancelled
     */
    public boolean isCancelled() {
        return errorCode == -6 || getMessage().contains("cancelled");
    }
}
