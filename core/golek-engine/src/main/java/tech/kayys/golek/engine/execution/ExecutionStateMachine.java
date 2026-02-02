package tech.kayys.golek.engine.execution;

/**
 * Deterministic state machine for execution lifecycle.
 * Pure function: (current state, signal) -> next state
 */
public interface ExecutionStateMachine {

    /**
     * Compute next state based on current state and signal
     */
    ExecutionStatus next(ExecutionStatus current, ExecutionSignal signal);

    /**
     * Validate if transition is allowed
     */
    boolean isTransitionAllowed(ExecutionStatus from, ExecutionStatus to);
}