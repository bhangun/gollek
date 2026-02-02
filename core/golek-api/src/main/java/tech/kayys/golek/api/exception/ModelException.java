package tech.kayys.golek.api.exception;

import tech.kayys.golek.api.error.ErrorCode;

/**
 * Exception thrown for model-related errors.
 */
public class ModelException extends RuntimeException {
    private final ErrorCode errorCode;
    private final String modelId;

    public ModelException(ErrorCode errorCode, String message, String modelId) {
        super(message);
        this.errorCode = errorCode;
        this.modelId = modelId;
    }

    public ModelException(ErrorCode errorCode, String message, String modelId, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
        this.modelId = modelId;
    }

    public ErrorCode getErrorCode() {
        return errorCode;
    }

    public String getModelId() {
        return modelId;
    }
}
