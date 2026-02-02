package tech.kayys.golek.provider.core.exception;

import tech.kayys.golek.api.exception.ProviderException;

/**
 * Exception thrown when no compatible provider found
 */
public class NoCompatibleProviderException extends ProviderException {
    public NoCompatibleProviderException(String message) {
        super(message);
    }
}
