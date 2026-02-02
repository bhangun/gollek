package tech.kayys.golek.engine.exception;

import tech.kayys.golek.api.error.ErrorCode;

/**
 * Exception thrown during runner initialization.
 *
 * @author bhangun
 * @since 1.0.0
 */
public class RunnerInitializationException extends InferenceException {

    /**
     * Creates a new RunnerInitializationException with the specified error code and
     * message.
     *
     * @param errorCode The error code associated with the exception
     * @param message   The error message
     */
    public RunnerInitializationException(ErrorCode errorCode, String message) {
        super(errorCode, message);
    }

    /**
     * Creates a new RunnerInitializationException with the specified error code,
     * message, and cause.
     *
     * @param errorCode The error code associated with the exception
     * @param message   The error message
     * @param cause     The underlying cause of the exception
     */
    public RunnerInitializationException(ErrorCode errorCode, String message, Throwable cause) {
        super(errorCode, message, cause);
    }

    /**
     * Creates a new RunnerInitializationException with additional context
     * information.
     *
     * @param errorCode The error code associated with the exception
     * @param message   The error message
     * @param context   Additional context information about the error
     */
    public RunnerInitializationException(ErrorCode errorCode, String message, java.util.Map<String, Object> context) {
        super(errorCode, message);
        if (context != null) {
            context.forEach(this::addContext);
        }
    }

    /**
     * Creates a new RunnerInitializationException with cause and additional context
     * information.
     *
     * @param errorCode The error code associated with the exception
     * @param message   The error message
     * @param cause     The underlying cause of the exception
     * @param context   Additional context information about the error
     */
    public RunnerInitializationException(ErrorCode errorCode, String message, Throwable cause,
            java.util.Map<String, Object> context) {
        super(errorCode, message, cause);
        if (context != null) {
            context.forEach(this::addContext);
        }
    }
}