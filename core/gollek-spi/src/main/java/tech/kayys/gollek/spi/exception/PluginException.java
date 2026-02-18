package tech.kayys.gollek.spi.exception;

/**
 * Base exception for plugin-related errors.
 */
public class PluginException extends RuntimeException {

    public PluginException(String message) {
        super(message);
    }

    public PluginException(String message, Throwable cause) {
        super(message, cause);
    }
}
