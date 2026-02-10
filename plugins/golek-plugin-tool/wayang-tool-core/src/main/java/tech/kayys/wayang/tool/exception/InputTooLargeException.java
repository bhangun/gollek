package tech.kayys.wayang.tool.exception;

/**
 * Exception thrown when the input size exceeds the limit.
 */
public class InputTooLargeException extends RuntimeException {
    public InputTooLargeException(String message) {
        super(message);
    }
}
