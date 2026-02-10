package tech.kayys.gamelan.core.engine;

import tech.kayys.gamelan.core.domain.*;

import java.util.Map;

/**
 * Result from executor after task execution
 */
public record NodeExecutionResult(
        WorkflowRunId runId,
        NodeId nodeId,
        int attempt,
        NodeExecutionStatus status,
        Map<String, Object> output,
        ErrorInfo error,
        ExecutionToken executionToken) {
    /**
     * Create successful result
     */
    public static NodeExecutionResult success(
            WorkflowRunId runId,
            NodeId nodeId,
            int attempt,
            Map<String, Object> output,
            ExecutionToken token) {

        return new NodeExecutionResult(
                runId,
                nodeId,
                attempt,
                NodeExecutionStatus.COMPLETED,
                output,
                null,
                token);
    }

    /**
     * Create failure result
     */
    public static NodeExecutionResult failure(
            WorkflowRunId runId,
            NodeId nodeId,
            int attempt,
            ErrorInfo error,
            ExecutionToken token) {

        return new NodeExecutionResult(
                runId,
                nodeId,
                attempt,
                NodeExecutionStatus.FAILED,
                Map.of(),
                error,
                token);
    }

    /**
     * Check if successful
     */
    public boolean isSuccess() {
        return status == NodeExecutionStatus.COMPLETED;
    }

    /**
     * Check if failed
     */
    public boolean isFailed() {
        return status == NodeExecutionStatus.FAILED;
    }
}