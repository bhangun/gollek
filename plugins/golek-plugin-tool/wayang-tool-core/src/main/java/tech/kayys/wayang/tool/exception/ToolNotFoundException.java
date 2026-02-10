package tech.kayys.wayang.tool.exception;

public class ToolNotFoundException extends RuntimeException {
    public ToolNotFoundException(String message) {
        super(message);
    }
}
