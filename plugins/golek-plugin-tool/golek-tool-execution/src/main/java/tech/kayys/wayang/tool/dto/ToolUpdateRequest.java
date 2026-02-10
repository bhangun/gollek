package tech.kayys.wayang.tool.dto;

import java.util.Set;

public record ToolUpdateRequest(
        Boolean enabled,
        String description,
        Set<String> tags) {
}
