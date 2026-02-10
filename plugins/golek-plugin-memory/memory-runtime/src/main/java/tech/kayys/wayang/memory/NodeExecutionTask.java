package tech.kayys.gamelan.core.engine;

import tech.kayys.gamelan.core.domain.*;

import java.util.Map;

/**
 * Task sent to executor for execution
 */
public record NodeExecutionTask(
        WorkflowRunId runId,
        NodeId nodeId,
        int attempt,
        ExecutionToken token,
        Map<String, Object> context) {
    /**
     * Get value from context
     */
    public Object getContextValue(String key) {
        return context.get(key);
    }

    /**
     * Get value with default
     */
    @SuppressWarnings("unchecked")
    public <T> T getContextValue(String key, T defaultValue) {
        return (T) context.getOrDefault(key, defaultValue);
    }
}