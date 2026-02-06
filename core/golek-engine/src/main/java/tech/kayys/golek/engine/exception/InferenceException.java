package tech.kayys.golek.engine.exception;

import tech.kayys.golek.spi.error.ErrorCode;
import java.util.HashMap;
import java.util.Map;

public class InferenceException extends RuntimeException {

    private final ErrorCode errorCode;
    private final Map<String, Object> context = new HashMap<>();

    public InferenceException(ErrorCode errorCode, String message) {
        super(message);
        this.errorCode = errorCode != null ? errorCode : ErrorCode.INTERNAL_ERROR;
    }

    public InferenceException(ErrorCode errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode != null ? errorCode : ErrorCode.INTERNAL_ERROR;
    }

    public InferenceException addContext(String key, Object value) {
        this.context.put(key, value);
        return this;
    }

    public ErrorCode getErrorCode() {
        return errorCode;
    }

    public int getHttpStatusCode() {
        return errorCode.getHttpStatus();
    }

    public boolean isRetryable() {
        return errorCode.isRetryable();
    }

    public Map<String, Object> getContext() {
        return new HashMap<>(context);
    }
}
