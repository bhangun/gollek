package tech.kayys.wayang.tool.dto;

import java.util.Map;
import jakarta.validation.constraints.NotBlank;

public record OpenApiToolRequest(
    @NotBlank String namespace,
    @NotBlank String sourceType,
    @NotBlank String source,
    String authProfileId,
    Map<String, Object> guardrailsConfig
) {}
