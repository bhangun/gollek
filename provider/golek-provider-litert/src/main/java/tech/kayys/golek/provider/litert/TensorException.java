package tech.kayys.golek.provider.litert;

import tech.kayys.golek.core.exception.InferenceException;
import tech.kayys.golek.provider.core.spi.ErrorCode;

public class TensorException extends InferenceException {
    public TensorException(String message) {
        super(message);
    }

    public TensorException(String message, Throwable cause) {
        super(message, cause);
    }

    public TensorException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }

    public TensorException(ErrorCode tensorInvalidData, String string, String name) {
        // TODO Auto-generated constructor stub
    }
}
