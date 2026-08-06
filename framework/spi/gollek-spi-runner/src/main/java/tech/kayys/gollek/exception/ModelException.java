package tech.kayys.gollek.exception;

//import tech.kayys.gollek.engine.exception.InferenceException;
import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;

/**
 * Exception thrown for model-related errors in the Aljabr Engine.
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
