package tech.kayys.golek.engine.execution;

/**
 * Exception thrown when an illegal state transition is attempted.
 */
public class IllegalStateTransitionException extends RuntimeException {

    public IllegalStateTransitionException(String message) {
        super(message);
    }

    public IllegalStateTransitionException(String message, Throwable cause) {
        super(message, cause);
    }
}
