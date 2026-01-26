package tech.kayys.golek.kernel.execution;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.util.Map;
import java.util.Set;

/**
 * Default implementation of execution state machine.
 * Enforces valid state transitions.
 */
@ApplicationScoped
public class DefaultExecutionStateMachine implements ExecutionStateMachine {

    private static final Logger LOG = Logger.getLogger(DefaultExecutionStateMachine.class);

    // Valid transitions map for validation
    private static final Map<ExecutionStatus, Set<ExecutionStatus>> ALLOWED_TRANSITIONS = Map.of(
            ExecutionStatus.CREATED, Set.of(
                    ExecutionStatus.RUNNING,
                    ExecutionStatus.CANCELLED),
            ExecutionStatus.RUNNING, Set.of(
                    ExecutionStatus.WAITING,
                    ExecutionStatus.RETRYING,
                    ExecutionStatus.COMPLETED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.SUSPENDED,
                    ExecutionStatus.CANCELLED),
            ExecutionStatus.WAITING, Set.of(
                    ExecutionStatus.RUNNING,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED),
            ExecutionStatus.SUSPENDED, Set.of(
                    ExecutionStatus.RUNNING,
                    ExecutionStatus.CANCELLED),
            ExecutionStatus.RETRYING, Set.of(
                    ExecutionStatus.RUNNING,
                    ExecutionStatus.FAILED));

    @Override
    public ExecutionStatus next(ExecutionStatus current, ExecutionSignal signal) {
        ExecutionStatus nextState = computeNextState(current, signal);

        if (!isTransitionAllowed(current, nextState)) {
            throw new IllegalStateTransitionException(
                    String.format(
                            "Invalid transition from %s to %s via signal %s",
                            current, nextState, signal));
        }

        LOG.debugf("State transition: %s -> %s (signal: %s)", current, nextState, signal);
        return nextState;
    }

    private ExecutionStatus computeNextState(ExecutionStatus current, ExecutionSignal signal) {
        return switch (current) {

            case CREATED -> switch (signal) {
                case START -> ExecutionStatus.RUNNING;
                case CANCEL -> ExecutionStatus.CANCELLED;
                default -> current;
            };

            case RUNNING -> switch (signal) {
                case EXECUTION_SUCCESS -> ExecutionStatus.COMPLETED;
                case PHASE_FAILURE, EXECUTION_FAILURE -> ExecutionStatus.RETRYING;
                case WAIT_REQUESTED -> ExecutionStatus.WAITING;
                case CANCEL -> ExecutionStatus.CANCELLED;
                case COMPENSATE -> ExecutionStatus.COMPENSATED;
                default -> current;
            };

            case RETRYING -> switch (signal) {
                case START, RESUME -> ExecutionStatus.RUNNING;
                case RETRY_EXHAUSTED -> ExecutionStatus.FAILED;
                case CANCEL -> ExecutionStatus.CANCELLED;
                default -> current;
            };

            case WAITING -> switch (signal) {
                case APPROVED, RESUME -> ExecutionStatus.RUNNING;
                case REJECTED -> ExecutionStatus.FAILED;
                case CANCEL -> ExecutionStatus.CANCELLED;
                default -> current;
            };

            case SUSPENDED -> switch (signal) {
                case RESUME -> ExecutionStatus.RUNNING;
                case CANCEL -> ExecutionStatus.CANCELLED;
                default -> current;
            };

            case COMPENSATED -> switch (signal) {
                case COMPENSATION_DONE -> ExecutionStatus.COMPLETED;
                default -> current;
            };

            // Terminal states - no transitions
            case COMPLETED, FAILED, CANCELLED -> current;
        };
    }

    @Override
    public boolean isTransitionAllowed(ExecutionStatus from, ExecutionStatus to) {
        // Self-transition always allowed
        if (from == to) {
            return true;
        }

        // Terminal states cannot transition
        if (from.isTerminal()) {
            return false;
        }

        Set<ExecutionStatus> allowed = ALLOWED_TRANSITIONS.get(from);
        return allowed != null && allowed.contains(to);
    }
}