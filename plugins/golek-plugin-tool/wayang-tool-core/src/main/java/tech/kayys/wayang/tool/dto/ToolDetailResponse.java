package tech.kayys.wayang.tool.dto;

import java.util.Map;
import java.util.Set;

public record ToolDetailResponse(
        String toolId,
        String name,
        String description,
        Map<String, Object> inputSchema,
        Map<String, Object> outputSchema,
        Set<String> capabilities,
        String capabilityLevel,
        boolean enabled,
        boolean readOnly,
        long totalInvocations) {
}