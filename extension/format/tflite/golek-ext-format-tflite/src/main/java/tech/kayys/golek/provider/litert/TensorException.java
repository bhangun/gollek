package tech.kayys.golek.provider.litert;

import tech.kayys.golek.spi.error.ErrorCode;
import tech.kayys.golek.spi.exception.InferenceException;

public class TensorException extends InferenceException {
    public TensorException(String message) {
        super(ErrorCode.TENSOR_INVALID_DATA, message);
    }

    public TensorException(String message, Throwable cause) {
        super(ErrorCode.TENSOR_INVALID_DATA, message, cause);
    }

    public TensorException(ErrorCode errorCode, String message) {
        super(errorCode, message);
    }

    public TensorException(ErrorCode errorCode, String message, String tensorName) {
        super(errorCode, message);
        addContext("tensor", tensorName);
    }
}
