package tech.kayys.wayang.tool.dto;

import java.util.Map;

public record ToolExecutionResponse(
        String status,
        Map<String, Object> output,
        String error,
        String errorCode,
        Boolean retryable,
        long executionTimeMs) {
}
