package tech.kayys.golek.provider.core.mcp;

public class PromptNotFoundException extends RuntimeException {
    public PromptNotFoundException(String message) {
        super(message);
    }
}