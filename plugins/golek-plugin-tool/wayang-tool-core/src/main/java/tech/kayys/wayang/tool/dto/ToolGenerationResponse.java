package tech.kayys.wayang.mcp.dto;

import java.util.List;

public record ToolGenerationResponse(
        String sourceId,
        String namespace,
        int toolsGenerated,
        List<String> toolIds,
        List<String> warnings) {
}