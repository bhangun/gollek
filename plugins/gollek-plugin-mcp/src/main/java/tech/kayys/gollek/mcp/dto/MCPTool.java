package tech.kayys.gollek.mcp.dto;

import java.util.Map;

/**
 * Represents an MCP tool definition.
 */
public class MCPTool {

    private final String name;
    private final String description;
    private final Map<String, Object> inputSchema;

    private MCPTool(Builder builder) {
        this.name = builder.name;
        this.description = builder.description;
        this.inputSchema = builder.inputSchema;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public Map<String, Object> getInputSchema() {
        return inputSchema;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static MCPTool fromMap(Map<String, Object> data) {
        return builder()
                .name((String) data.get("name"))
                .description((String) data.get("description"))
                .inputSchema((Map<String, Object>) data.get("inputSchema"))
                .build();
    }

    public static class Builder {
        private String name;
        private String description;
        private Map<String, Object> inputSchema;

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder inputSchema(Map<String, Object> inputSchema) {
            this.inputSchema = inputSchema;
            return this;
        }

        public MCPTool build() {
            return new MCPTool(this);
        }
    }
}