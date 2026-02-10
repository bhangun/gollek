package tech.kayys.wayang.mcp.dto;

import java.util.Set;

public record ToolUpdateRequest(
        Boolean enabled,
        String description,
        Set<String> tags) {
}