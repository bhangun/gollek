package tech.kayys.golek.core.exception;

public class AuthorizationException extends InferenceException {
    
    public AuthorizationException(String message) {
        super(message);
    }
    
    public AuthorizationException(String message, Throwable cause) {
        super(message, cause);
    }
}