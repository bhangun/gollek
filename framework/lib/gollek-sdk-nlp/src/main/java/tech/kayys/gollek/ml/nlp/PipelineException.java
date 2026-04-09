package tech.kayys.gollek.ml.nlp;

/**
 * Unchecked exception thrown when a pipeline operation fails.
 *
 * <p>Wraps lower-level inference, I/O, or configuration errors so callers
 * can handle NLP failures without checked-exception boilerplate.
 *
 * @see Pipeline
 */
public class PipelineException extends RuntimeException {

    /**
     * Constructs a {@code PipelineException} with the given detail message.
     *
     * @param message human-readable description of the failure
     */
    public PipelineException(String message) {
        super(message);
    }

    /**
     * Constructs a {@code PipelineException} with a detail message and a root cause.
     *
     * @param message human-readable description of the failure
     * @param cause   the underlying exception that triggered this failure
     */
    public PipelineException(String message, Throwable cause) {
        super(message, cause);
    }
}
