package tech.kayys.golek.spi.execution;

/**
 * Canonical execution states for inference requests.
 * These states are the single source of truth.
 */
public enum ExecutionStatus {

    /**
     * Request created, not yet started
     */
    CREATED("Created", false, false),

    /**
     * Actively executing through phases
     */
    RUNNING("Running", false, false),

    /**
     * Waiting for external event (HITL, async callback)
     */
    WAITING("Waiting", false, false),

    /**
     * Paused by policy or manual intervention
     */
    SUSPENDED("Suspended", false, false),

    /**
     * In retry backoff period
     */
    RETRYING("Retrying", false, false),

    /**
     * Successfully completed
     */
    COMPLETED("Completed", true, false),

    /**
     * Terminal failure (exhausted retries)
     */
    FAILED("Failed", true, true),

    /**
     * Rollback/compensation completed
     */
    COMPENSATED("Compensated", true, false),

    /**
     * Cancelled by user/system
     */
    CANCELLED("Cancelled", true, false);

    private final String displayName;
    private final boolean terminal;
    private final boolean error;

    ExecutionStatus(String displayName, boolean terminal, boolean error) {
        this.displayName = displayName;
        this.terminal = terminal;
        this.error = error;
    }

    public String getDisplayName() {
        return displayName;
    }

    public boolean isTerminal() {
        return terminal;
    }

    public boolean isError() {
        return error;
    }

    public boolean isActive() {
        return !terminal;
    }

    public boolean canTransitionTo(ExecutionStatus target) {
        if (this == target) {
            return true;
        }
        if (this.isTerminal()) {
            return false;
        }

        return switch (this) {
            case CREATED -> target == RUNNING || target == CANCELLED;
            case RUNNING -> target == WAITING || target == RETRYING ||
                    target == COMPLETED || target == FAILED ||
                    target == SUSPENDED || target == CANCELLED;
            case WAITING -> target == RUNNING || target == FAILED ||
                    target == CANCELLED;
            case SUSPENDED -> target == RUNNING || target == CANCELLED;
            case RETRYING -> target == RUNNING || target == FAILED;
            default -> false;
        };
    }
}
