package tech.kayys.gollek.client.exception;

/**
 * Thrown when the server responds with HTTP 429 (Too Many Requests).
 *
 * <p>Carries the number of seconds the caller should wait before retrying,
 * sourced from the {@code Retry-After} response header when present.
 * Always carries the error code {@code "RATE_LIMIT_ERROR"}.
 *
 * @see GollekClientException
 */
public class RateLimitException extends GollekClientException {

    private final int retryAfterSeconds;

    /**
     * Constructs a rate-limit exception.
     *
     * @param message            description of the rate-limit failure
     * @param retryAfterSeconds  seconds to wait before retrying; sourced from {@code Retry-After} header
     */
    public RateLimitException(String message, int retryAfterSeconds) {
        super("RATE_LIMIT_ERROR", message);
        this.retryAfterSeconds = retryAfterSeconds;
    }

    /**
     * Constructs a rate-limit exception with a cause.
     *
     * @param message            description of the rate-limit failure
     * @param retryAfterSeconds  seconds to wait before retrying
     * @param cause              the underlying exception
     */
    public RateLimitException(String message, int retryAfterSeconds, Throwable cause) {
        super("RATE_LIMIT_ERROR", message, cause);
        this.retryAfterSeconds = retryAfterSeconds;
    }

    /**
     * Returns the number of seconds the caller should wait before retrying.
     *
     * @return retry delay in seconds (defaults to 60 if the header was absent)
     */
    public int getRetryAfterSeconds() {
        return retryAfterSeconds;
    }
}