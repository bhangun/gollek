package tech.kayys.golek.model.exception;

import tech.kayys.golek.spi.error.ErrorCode;

/**
 * Exception thrown when model repository operations fail.
 */
public class InferenceException extends tech.kayys.golek.spi.exception.InferenceException {

    private static final long serialVersionUID = 1L;

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
