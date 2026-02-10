package tech.kayys.wayang.tool.spi;

/**
 * Exception thrown when tool validation fails.
 */
public class ToolValidationException extends RuntimeException {

    public ToolValidationException(String message) {
        super(message);
    }

    public ToolValidationException(String message, Throwable cause) {
        super(message, cause);
    }
}