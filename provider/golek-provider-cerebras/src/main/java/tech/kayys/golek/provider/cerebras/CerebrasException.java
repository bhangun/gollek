package tech.kayys.golek.provider.cerebras;

import tech.kayys.golek.spi.exception.ProviderException;

/**
 * Exception thrown when Cerebras API returns an error.
 */
public class CerebrasException extends ProviderException {
    public CerebrasException(String message) {
        super(message);
    }

    public CerebrasException(String message, Throwable cause) {
        super(message, cause);
    }
}