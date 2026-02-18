package tech.kayys.gollek.mcp.dto;

import java.util.Map;

/**
 * Represents an MCP resource.
 */
public class MCPResource {
    private final String uri;
    private final String name;
    private final String description;
    private final Map<String, Object> properties;

    private MCPResource(Builder builder) {
        this.uri = builder.uri;
        this.name = builder.name;
        this.description = builder.description;
        this.properties = builder.properties != null ? builder.properties : Map.of();
    }

    public String getUri() {
        return uri;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public String getMimeType() {
        return (String) properties.get("mimeType");
    }

    public Map<String, Object> getProperties() {
        return properties;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static MCPResource fromMap(Map<String, Object> data) {
        return builder()
                .uri((String) data.get("uri"))
                .name((String) data.get("name"))
                .description((String) data.get("description"))
                .properties((Map<String, Object>) data.get("properties"))
                .build();
    }

    public static class Builder {
        private String uri;
        private String name;
        private String description;
        private Map<String, Object> properties;

        public Builder uri(String uri) {
            this.uri = uri;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder properties(Map<String, Object> properties) {
            this.properties = properties;
            return this;
        }

        public MCPResource build() {
            return new MCPResource(this);
        }
    }
}