package tech.kayys.gollek.provider.cerebras;

import tech.kayys.gollek.spi.exception.ProviderException;

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