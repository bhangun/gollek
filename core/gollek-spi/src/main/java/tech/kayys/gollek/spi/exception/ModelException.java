package tech.kayys.gollek.spi.exception;

import tech.kayys.gollek.spi.error.ErrorCode;

/**
 * Exception thrown for model-related errors.
 */
public class ModelException extends InferenceException {
    private final String modelId;

    public ModelException(ErrorCode errorCode, String message, String modelId) {
        super(errorCode, message);
        this.modelId = modelId;
    }

    public ModelException(ErrorCode errorCode, String message, String modelId, Throwable cause) {
        super(errorCode, message, cause);
        this.modelId = modelId;
    }

    public String getModelId() {
        return modelId;
    }
}
