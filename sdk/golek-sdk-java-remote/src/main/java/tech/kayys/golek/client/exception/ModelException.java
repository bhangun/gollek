package tech.kayys.golek.client.exception;

/**
 * Exception for model-related errors.
 */
public class ModelException extends GolekClientException {
    
    private final String modelId;
    
    public ModelException(String modelId, String message) {
        super("MODEL_ERROR", message);
        this.modelId = modelId;
    }
    
    public ModelException(String modelId, String message, Throwable cause) {
        super("MODEL_ERROR", message, cause);
        this.modelId = modelId;
    }
    
    public String getModelId() {
        return modelId;
    }
}