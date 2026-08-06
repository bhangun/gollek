package tech.kayys.gollek.runtime.control;

public enum ExecutionState {
    RUNNING,
    PAUSED,
    CANCELLED,
    COMPLETED,
    FAILED,
    SHUTDOWN
}