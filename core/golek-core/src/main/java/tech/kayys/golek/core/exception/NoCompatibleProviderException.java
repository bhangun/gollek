package tech.kayys.golek.core.exception;

public class NoCompatibleProviderException extends InferenceException {
    
    public NoCompatibleProviderException(String message) {
        super(message);
    }
    
    public NoCompatibleProviderException(String message, Throwable cause) {
        super(message, cause);
    }
}