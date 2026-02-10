package tech.kayys.wayang.tool.dto;

import lombok.Data;

@Data
public class OpenApiToolRequest {
    private String namespace;
    private String url;
    private String sourceType;
    private String source;
    private String authProfileId;
    private java.util.Map<String, Object> guardrailsConfig;
}
