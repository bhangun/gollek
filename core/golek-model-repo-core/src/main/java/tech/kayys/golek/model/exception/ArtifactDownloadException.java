package tech.kayys.golek.model.exception;

import tech.kayys.golek.spi.error.ErrorCode;

/**
 * Exception thrown when model artifact download fails.
 */
public class ArtifactDownloadException extends Exception {

    private static final long serialVersionUID = 1L;

    private final ErrorCode errorCode;

    public ArtifactDownloadException(String message) {
        this(ErrorCode.NETWORK_BAD_RESPONSE, message);
    }

    public ArtifactDownloadException(String message, Throwable cause) {
        this(ErrorCode.NETWORK_BAD_RESPONSE, message, cause);
    }

    public ArtifactDownloadException(ErrorCode errorCode, String message) {
        super(message);
        this.errorCode = errorCode != null ? errorCode : ErrorCode.INTERNAL_ERROR;
    }

    public ArtifactDownloadException(ErrorCode errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode != null ? errorCode : ErrorCode.INTERNAL_ERROR;
    }

    public ErrorCode getErrorCode() {
        return errorCode;
    }
}
