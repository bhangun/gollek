package tech.kayys.golek.model.exception;

/**
 * Exception thrown when model artifact download fails
 */
public class ArtifactDownloadException extends Exception {

    private static final long serialVersionUID = 1L;

    public ArtifactDownloadException(String message) {
        super(message);
    }

    public ArtifactDownloadException(String message, Throwable cause) {
        super(message, cause);
    }
}
