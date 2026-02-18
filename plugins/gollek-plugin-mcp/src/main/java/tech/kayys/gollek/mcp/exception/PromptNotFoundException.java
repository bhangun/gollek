package tech.kayys.gollek.mcp.exception;

public class PromptNotFoundException extends RuntimeException {
    public PromptNotFoundException(String message) {
        super(message);
    }
}