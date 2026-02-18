package tech.kayys.gollek.client.exception;

/**
 * Exception for authentication errors.
 */
public class AuthenticationException extends GollekClientException {
    
    public AuthenticationException(String message) {
        super("AUTH_ERROR", message);
    }
    
    public AuthenticationException(String message, Throwable cause) {
        super("AUTH_ERROR", message, cause);
    }
}