package tech.kayys.golek.model.exception;

/**
 * Exception thrown when model not found
 */
public class ModelNotFoundException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public ModelNotFoundException(String message) {
        super(message);
    }
}