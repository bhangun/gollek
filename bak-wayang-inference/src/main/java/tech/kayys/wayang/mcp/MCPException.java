package tech.kayys.golek.mcp;


public class MCPException extends Exception {
    public MCPException(String message) {
        super(message);
    }
    
    public MCPException(String message, Throwable cause) {
        super(message, cause);
    }
}
