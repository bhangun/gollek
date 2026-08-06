package tech.kayys.gollek.spi.execution;

import tech.kayys.gollek.error.ErrorCode;

/**
 * Compatibility wrapper for InferenceException in execution package.
 * Extends the canonical SPI exception.
 */
public class InferenceException extends tech.kayys.gollek.spi.exception.InferenceException {

    public InferenceException(ErrorCode errorCode, String message) {
        super(errorCode, message);
    }

    public InferenceException(ErrorCode errorCode, String message, Throwable cause) {
        super(errorCode, message, cause);
    }

    public InferenceException(String message) {
        super(ErrorCode.INTERNAL_ERROR, message);
    }

    public InferenceException(String message, Throwable cause) {
        super(ErrorCode.INTERNAL_ERROR, message, cause);
    }
}
