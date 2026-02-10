package tech.kayys.wayang.mcp.dto;

import java.util.List;
import java.util.UUID;

public record ToolGenerationResult(
    UUID sourceId,
    String namespace,
    int toolsGenerated,
    List<String> toolIds,
    List<String> warnings
) {}