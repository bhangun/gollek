package tech.kayys.golek.core.exception;

/**
 * Exception thrown when an inference error occurs.
 */
public class InferenceException extends RuntimeException {

    public InferenceException(String message) {
        super(message);
    }

    public InferenceException(String message, Throwable cause) {
        super(message, cause);
    }
}