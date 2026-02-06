package tech.kayys.golek.core.exception;

import tech.kayys.golek.spi.error.ErrorCode;

public class NoCompatibleProviderException extends InferenceException {
    
    public NoCompatibleProviderException(String message) {
        super(ErrorCode.ROUTING_NO_COMPATIBLE_PROVIDER, message);
    }
    
    public NoCompatibleProviderException(String message, Throwable cause) {
        super(ErrorCode.ROUTING_NO_COMPATIBLE_PROVIDER, message, cause);
    }
}
