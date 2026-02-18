package tech.kayys.gollek.sdk.local;

/**
 * Runtime exception wrapper for local SDK client operations.
 */
public class GollekLocalClientException extends RuntimeException {

    public GollekLocalClientException(String message, Throwable cause) {
        super(message, cause);
    }
}
