package tech.kayys.gollek.sdk.core.exception;

/**
 * Base exception for the Gollek SDK.
 */
public class SdkException extends Exception {
    
    private final String errorCode;
    
    public SdkException(String message) {
        super(message);
        this.errorCode = "SDK_ERROR";
    }
    
    public SdkException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = "SDK_ERROR";
    }
    
    public SdkException(String errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }
    
    public SdkException(String errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }
    
    public String getErrorCode() {
        return errorCode;
    }
}