package tech.kayys.golek.core.exception;

import tech.kayys.golek.spi.error.ErrorCode;

public class AllRunnersFailedException extends InferenceException {
    
    public AllRunnersFailedException(String message) {
        super(ErrorCode.ALL_RUNNERS_FAILED, message);
    }
    
    public AllRunnersFailedException(String message, Throwable cause) {
        super(ErrorCode.ALL_RUNNERS_FAILED, message, cause);
    }
}
