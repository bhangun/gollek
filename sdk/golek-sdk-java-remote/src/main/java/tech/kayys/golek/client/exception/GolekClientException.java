package tech.kayys.golek.client.exception;

/**
 * Exception thrown when there's an error in the Golek client operations.
 */
public class GolekClientException extends Exception {
    
    private final String errorCode;
    
    public GolekClientException(String message) {
        super(message);
        this.errorCode = "CLIENT_ERROR";
    }
    
    public GolekClientException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = "CLIENT_ERROR";
    }
    
    public GolekClientException(String errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }
    
    public GolekClientException(String errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }
    
    public String getErrorCode() {
        return errorCode;
    }
}