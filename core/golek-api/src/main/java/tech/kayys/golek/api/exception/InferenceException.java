package tech.kayys.golek.api.exception;

import tech.kayys.golek.api.error.ErrorCode;
import java.util.HashMap;
import java.util.Map;

public class InferenceException extends RuntimeException {

    private final ErrorCode errorCode;
    private final Map<String, Object> context = new HashMap<>();

    public InferenceException(ErrorCode errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }

    public InferenceException(String message) {
        this(ErrorCode.INTERNAL_ERROR, message);
    }

    public InferenceException(ErrorCode errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }

    public InferenceException(String message, Throwable cause) {
        this(ErrorCode.INTERNAL_ERROR, message, cause);
    }

    public InferenceException addContext(String key, Object value) {
        this.context.put(key, value);
        return this;
    }

    public ErrorCode getErrorCode() {
        return errorCode;
    }

    public Map<String, Object> getContext() {
        return new HashMap<>(context);
    }
}
