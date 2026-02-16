package tech.kayys.wayang.tool.dto;

import java.util.Map;

public record GenerateToolsRequest(
                String requestId,
                String namespace,
                SourceType sourceType,
                String source,
                String authProfileId,
                String userId,
                Map<String, Object> guardrailsConfig) {
}