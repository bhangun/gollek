package tech.kayys.golek.client.exception;

/**
 * Exception for authentication errors.
 */
public class AuthenticationException extends GolekClientException {
    
    public AuthenticationException(String message) {
        super("AUTH_ERROR", message);
    }
    
    public AuthenticationException(String message, Throwable cause) {
        super("AUTH_ERROR", message, cause);
    }
}