package tech.kayys.gollek.mcp.dto;

import java.util.List;
import java.util.Map;

/**
 * Represents an MCP prompt.
 */
public class MCPPrompt {
    private final String name;
    private final String description;
    private final List<String> arguments;
    private final Map<String, Object> properties;

    private MCPPrompt(Builder builder) {
        this.name = builder.name;
        this.description = builder.description;
        this.arguments = builder.arguments;
        this.properties = builder.properties != null ? builder.properties : Map.of();
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public List<String> getArguments() {
        return arguments;
    }

    public Map<String, Object> getProperties() {
        return properties;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static MCPPrompt fromMap(Map<String, Object> data) {
        return builder()
                .name((String) data.get("name"))
                .description((String) data.get("description"))
                .arguments((List<String>) data.get("arguments"))
                .properties((Map<String, Object>) data.get("properties"))
                .build();
    }

    public static class Builder {
        private String name;
        private String description;
        private List<String> arguments;
        private Map<String, Object> properties;

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder arguments(List<String> arguments) {
            this.arguments = arguments;
            return this;
        }

        public Builder properties(Map<String, Object> properties) {
            this.properties = properties;
            return this;
        }

        public MCPPrompt build() {
            return new MCPPrompt(this);
        }
    }
}