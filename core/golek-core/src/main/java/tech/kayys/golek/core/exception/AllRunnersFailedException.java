package tech.kayys.golek.core.exception;

public class AllRunnersFailedException extends InferenceException {
    
    public AllRunnersFailedException(String message) {
        super(message);
    }
    
    public AllRunnersFailedException(String message, Throwable cause) {
        super(message, cause);
    }
}