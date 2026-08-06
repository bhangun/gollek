package tech.kayys.gollek.client.exception;

/**
 * Thrown when the server rejects a request due to invalid or missing credentials
 * (HTTP 401 or 403).
 *
 * <p>Always carries the error code {@code "AUTH_ERROR"}.
 *
 * @see GollekClientException
 */
public class AuthenticationException extends GollekClientException {

    /**
     * Constructs an authentication exception with the given message.
     *
     * @param message description of the authentication failure
     */
    public AuthenticationException(String message) {
        super("AUTH_ERROR", message);
    }

    /**
     * Constructs an authentication exception with a message and cause.
     *
     * @param message description of the authentication failure
     * @param cause   the underlying exception
     */
    public AuthenticationException(String message, Throwable cause) {
        super("AUTH_ERROR", message, cause);
    }
}