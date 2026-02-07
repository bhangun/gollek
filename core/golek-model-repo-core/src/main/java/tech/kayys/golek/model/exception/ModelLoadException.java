package tech.kayys.golek.model.exception;

import tech.kayys.golek.spi.error.ErrorCode;
import tech.kayys.golek.spi.exception.ModelException;

public class ModelLoadException extends ModelException {

    public ModelLoadException(String message) {
        super(ErrorCode.INIT_MODEL_LOAD_FAILED, message, null);
    }

    public ModelLoadException(String message, Throwable cause) {
        super(ErrorCode.INIT_MODEL_LOAD_FAILED, message, null, cause);
    }

    public ModelLoadException(ErrorCode errorCode, String message, String modelId) {
        super(errorCode, message, modelId);
    }

    public ModelLoadException(ErrorCode errorCode, String message, String modelId, Throwable cause) {
        super(errorCode, message, modelId, cause);
    }
}
