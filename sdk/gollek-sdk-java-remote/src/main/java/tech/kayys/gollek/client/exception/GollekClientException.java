package tech.kayys.gollek.client.exception;

/**
 * Exception thrown when a Gollek client HTTP operation fails.
 *
 * <p>Carries a machine-readable {@link #getErrorCode() error code} alongside the
 * human-readable message so callers can distinguish error categories without
 * parsing message strings.
 *
 * @see AuthenticationException
 * @see RateLimitException
 * @see ModelException
 */
public class GollekClientException extends Exception {

    private final String errorCode;

    /**
     * Constructs an exception with the default error code {@code "CLIENT_ERROR"}.
     *
     * @param message human-readable description of the failure
     */
    public GollekClientException(String message) {
        super(message);
        this.errorCode = "CLIENT_ERROR";
    }

    /**
     * Constructs an exception with the default error code {@code "CLIENT_ERROR"} and a cause.
     *
     * @param message human-readable description of the failure
     * @param cause   the underlying exception
     */
    public GollekClientException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = "CLIENT_ERROR";
    }

    /**
     * Constructs an exception with an explicit error code.
     *
     * @param errorCode machine-readable error identifier (e.g. {@code "AUTH_ERROR"})
     * @param message   human-readable description of the failure
     */
    public GollekClientException(String errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }

    /**
     * Constructs an exception with an explicit error code and a cause.
     *
     * @param errorCode machine-readable error identifier
     * @param message   human-readable description of the failure
     * @param cause     the underlying exception
     */
    public GollekClientException(String errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }

    /**
     * Returns the machine-readable error code for this exception.
     *
     * @return error code string, never {@code null}
     */
    public String getErrorCode() {
        return errorCode;
    }
}