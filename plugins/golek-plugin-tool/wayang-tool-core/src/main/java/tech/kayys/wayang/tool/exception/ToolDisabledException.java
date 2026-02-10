package tech.kayys.wayang.tool.exception;

/**
 * Exception thrown when a tool is disabled.
 */
public class ToolDisabledException extends RuntimeException {
    public ToolDisabledException(String message) {
        super(message);
    }
}
