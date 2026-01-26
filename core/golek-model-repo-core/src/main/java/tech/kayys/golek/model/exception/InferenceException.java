package tech.kayys.golek.model.exception;

/**
 * Exception thrown when inference fails
 */
public class InferenceException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public InferenceException(String message) {
        super(message);
    }

    public InferenceException(String message, Throwable cause) {
        super(message, cause);
    }
}
