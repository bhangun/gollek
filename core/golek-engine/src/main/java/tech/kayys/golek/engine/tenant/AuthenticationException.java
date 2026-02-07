package tech.kayys.golek.engine.tenant;

import tech.kayys.golek.spi.exception.InferenceException;
import tech.kayys.golek.spi.error.ErrorCode;

public class AuthenticationException extends InferenceException {

    public AuthenticationException(ErrorCode errorCode, String message) {
        super(errorCode, message);
    }
}
