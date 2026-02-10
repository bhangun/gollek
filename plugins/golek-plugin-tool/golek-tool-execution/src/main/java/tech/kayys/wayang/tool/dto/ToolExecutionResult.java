package tech.kayys.wayang.tool.dto;

import java.util.Map;

/**
 * Tool execution result
 */
public record ToolExecutionResult(
        String toolId,
        InvocationStatus status,
        Map<String, Object> output,
        String errorMessage,
        long executionTimeMs,
        Map<String, Object> metadata) {
    public static ToolExecutionResult success(
            String toolId,
            Map<String, Object> output,
            long executionTimeMs) {
        return new ToolExecutionResult(
                toolId,
                InvocationStatus.SUCCESS,
                output,
                null,
                executionTimeMs,
                Map.of());
    }

    public static ToolExecutionResult failure(
            String toolId,
            InvocationStatus status,
            String errorMessage,
            long executionTimeMs) {
        return new ToolExecutionResult(
                toolId,
                status,
                Map.of(),
                errorMessage,
                executionTimeMs,
                Map.of());
    }

    public static ToolExecutionResult failure(
            String toolId,
            InvocationStatus status,
            String errorMessage,
            long executionTimeMs,
            Map<String, Object> metadata) {
        return new ToolExecutionResult(
                toolId,
                status,
                Map.of(),
                errorMessage,
                executionTimeMs,
                metadata != null ? metadata : Map.of());
    }
}
