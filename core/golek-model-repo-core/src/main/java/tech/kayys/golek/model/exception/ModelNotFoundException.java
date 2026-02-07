package tech.kayys.golek.model.exception;

import tech.kayys.golek.spi.error.ErrorCode;
import tech.kayys.golek.spi.exception.ModelException;

/**
 * Exception thrown when model not found.
 */
public class ModelNotFoundException extends ModelException {

    private static final long serialVersionUID = 1L;

    public ModelNotFoundException(String message) {
        super(ErrorCode.MODEL_NOT_FOUND, message, null);
    }

    public ModelNotFoundException(String message, String modelId) {
        super(ErrorCode.MODEL_NOT_FOUND, message, modelId);
    }

    public ModelNotFoundException(String message, String modelId, Throwable cause) {
        super(ErrorCode.MODEL_NOT_FOUND, message, modelId, cause);
    }
}
