package tech.kayys.wayang.tool.dto;

import java.util.Map;
import java.util.Set;

/**
 * Tool metadata for agent discovery
 */
public record ToolMetadata(
        String toolId,
        String name,
        String description,
        Map<String, Object> inputSchema,
        Set<String> capabilities,
        CapabilityLevel capabilityLevel,
        boolean readOnly) {
}
