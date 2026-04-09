package tech.kayys.gollek.client.exception;

/**
 * Thrown when the server returns a model-related error (HTTP 422 or a
 * model-specific failure).
 *
 * <p>Carries the {@code modelId} that triggered the error when known.
 * Always carries the error code {@code "MODEL_ERROR"}.
 *
 * @see GollekClientException
 */
public class ModelException extends GollekClientException {

    private final String modelId;

    /**
     * Constructs a model exception.
     *
     * @param modelId the model identifier that caused the error, or {@code null} if unknown
     * @param message description of the failure
     */
    public ModelException(String modelId, String message) {
        super("MODEL_ERROR", message);
        this.modelId = modelId;
    }

    /**
     * Constructs a model exception with a cause.
     *
     * @param modelId the model identifier that caused the error, or {@code null} if unknown
     * @param message description of the failure
     * @param cause   the underlying exception
     */
    public ModelException(String modelId, String message, Throwable cause) {
        super("MODEL_ERROR", message, cause);
        this.modelId = modelId;
    }

    /**
     * Returns the model identifier associated with this error.
     *
     * @return model ID, or {@code null} if not available
     */
    public String getModelId() {
        return modelId;
    }
}