package tech.kayys.gollek.litert;

/**
 * LiteRT exception.
 */
public class LiteRTException extends RuntimeException {

    public LiteRTException(String message) {
        super(message);
    }

    public LiteRTException(String message, Throwable cause) {
        super(message, cause);
    }
}
