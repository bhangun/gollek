package tech.kayys.gollek.client.exception;

/**
 * Exception thrown when there's an error in the Gollek client operations.
 */
public class GollekClientException extends Exception {
    
    private final String errorCode;
    
    public GollekClientException(String message) {
        super(message);
        this.errorCode = "CLIENT_ERROR";
    }
    
    public GollekClientException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = "CLIENT_ERROR";
    }
    
    public GollekClientException(String errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }
    
    public GollekClientException(String errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }
    
    public String getErrorCode() {
        return errorCode;
    }
}