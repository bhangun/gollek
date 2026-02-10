package tech.kayys.golek.tool.dto;

import java.util.Map;

/**
 * Tool execution result
 */
public record ToolExecutionResult(
        String toolCallId,
        String toolName,
        InvocationStatus status,
        Map<String, Object> output,
        String errorMessage,
        long executionTimeMs,
        Map<String, Object> metadata,
        boolean success) {
    
    public static ToolExecutionResult success(
            String toolCallId,
            String toolName,
            Map<String, Object> output,
            long executionTimeMs) {
        return new ToolExecutionResult(
                toolCallId,
                toolName,
                InvocationStatus.SUCCESS,
                output,
                null,
                executionTimeMs,
                Map.of(),
                true);
    }

    public static ToolExecutionResult failure(
            String toolCallId,
            String toolName,
            String errorMessage,
            long executionTimeMs) {
        return new ToolExecutionResult(
                toolCallId,
                toolName,
                InvocationStatus.FAILURE,
                Map.of(),
                errorMessage,
                executionTimeMs,
                Map.of(),
                false);
    }

    public static ToolExecutionResult failure(
            String toolCallId,
            String toolName,
            String errorMessage,
            long executionTimeMs,
            Map<String, Object> metadata) {
        return new ToolExecutionResult(
                toolCallId,
                toolName,
                InvocationStatus.FAILURE,
                Map.of(),
                errorMessage,
                executionTimeMs,
                metadata != null ? metadata : Map.of(),
                false);
    }
}