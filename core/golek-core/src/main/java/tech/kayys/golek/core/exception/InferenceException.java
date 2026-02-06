package tech.kayys.golek.core.exception;

import tech.kayys.golek.spi.error.ErrorCode;

/**
 * Exception thrown when an inference error occurs.
 */
public class InferenceException extends tech.kayys.golek.spi.exception.InferenceException {

    public InferenceException(String message) {
        super(ErrorCode.INTERNAL_ERROR, message);
    }

    public InferenceException(String message, Throwable cause) {
        super(ErrorCode.INTERNAL_ERROR, message, cause);
    }

    public InferenceException(ErrorCode errorCode, String message) {
        super(errorCode, message);
    }

    public InferenceException(ErrorCode errorCode, String message, Throwable cause) {
        super(errorCode, message, cause);
    }
}
