package tech.kayys.golek.engine.execution;

/**
 * Signals that trigger state transitions.
 * These are events, not states.
 */
public enum ExecutionSignal {

    /**
     * Execution started
     */
    START,

    /**
     * Phase completed successfully
     */
    PHASE_SUCCESS,

    /**
     * Phase failed
     */
    PHASE_FAILURE,

    /**
     * All phases completed
     */
    EXECUTION_SUCCESS,

    /**
     * Execution failed
     */
    EXECUTION_FAILURE,

    /**
     * Retry limit exhausted
     */
    RETRY_EXHAUSTED,

    /**
     * External wait requested (HITL, async)
     */
    WAIT_REQUESTED,

    /**
     * External approval received
     */
    APPROVED,

    /**
     * External rejection received
     */
    REJECTED,

    /**
     * Compensation triggered
     */
    COMPENSATE,

    /**
     * Compensation completed
     */
    COMPENSATION_DONE,

    /**
     * Cancellation requested
     */
    CANCEL,

    /**
     * Resume from suspended state
     */
    RESUME
}