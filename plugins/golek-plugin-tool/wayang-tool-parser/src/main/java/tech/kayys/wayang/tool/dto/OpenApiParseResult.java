package tech.kayys.wayang.tool.dto;

import io.swagger.v3.oas.models.OpenAPI;
import java.util.List;

public record OpenApiParseResult(
    OpenAPI openApi,
    String rawSpec,
    boolean isValid,
    List<String> errors
) {}