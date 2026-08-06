package tech.kayys.gollek.model.exception;

/** Exception thrown when a model cannot be loaded or initialized. */
public class ModelLoadException extends RuntimeException {
    private final String modelId;

    public ModelLoadException(String modelId, String message) {
        super("Model [" + modelId + "]: " + message);
        this.modelId = modelId;
    }

    public ModelLoadException(String modelId, String message, Throwable cause) {
        super("Model [" + modelId + "]: " + message, cause);
        this.modelId = modelId;
    }

    public String getModelId() { return modelId; }
}
