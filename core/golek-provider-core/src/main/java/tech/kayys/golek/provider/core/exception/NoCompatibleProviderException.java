package tech.kayys.golek.provider.core.exception;

/**
 * Exception thrown when no compatible provider found
 */
public class NoCompatibleProviderException extends RuntimeException {
    public NoCompatibleProviderException(String message) {
        super(message);
    }
}
