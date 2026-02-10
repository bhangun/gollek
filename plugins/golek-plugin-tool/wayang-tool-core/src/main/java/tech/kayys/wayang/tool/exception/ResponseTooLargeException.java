package tech.kayys.wayang.tool.exception;

/**
 * Exception thrown when the tool execution response size exceeds the configured
 * limit.
 */
public class ResponseTooLargeException extends RuntimeException {

    private final long size;
    private final long limit;

    public ResponseTooLargeException(String message) {
        this(message, 0, 0);
    }

    public ResponseTooLargeException(String message, long size, long limit) {
        super(message);
        this.size = size;
        this.limit = limit;
    }

    public long getSize() {
        return size;
    }

    public long getLimit() {
        return limit;
    }
}
