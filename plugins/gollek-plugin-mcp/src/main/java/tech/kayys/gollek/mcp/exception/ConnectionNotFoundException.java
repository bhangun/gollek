package tech.kayys.gollek.mcp.exception;

/**
 * Thrown when an MCP connection is not found.
 */
public class ConnectionNotFoundException extends RuntimeException {
    public ConnectionNotFoundException(String message) {
        super(message);
    }

    public ConnectionNotFoundException(String message, Throwable cause) {
        super(message, cause);
    }
}