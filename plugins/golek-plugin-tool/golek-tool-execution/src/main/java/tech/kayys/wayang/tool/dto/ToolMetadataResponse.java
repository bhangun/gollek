package tech.kayys.wayang.tool.dto;

import java.util.Set;

public record ToolMetadataResponse(
        String toolId,
        String name,
        String description,
        Set<String> capabilities,
        String capabilityLevel,
        boolean readOnly,
        Set<String> tags) {
}
