package tech.kayys.golek.core.exception;

import tech.kayys.golek.spi.error.ErrorCode;

public class AuthorizationException extends InferenceException {
    
    public AuthorizationException(String message) {
        super(ErrorCode.AUTH_PERMISSION_DENIED, message);
    }
    
    public AuthorizationException(String message, Throwable cause) {
        super(ErrorCode.AUTH_PERMISSION_DENIED, message, cause);
    }
}
