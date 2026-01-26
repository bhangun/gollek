package tech.kayys.golek.mcp;


import com.fasterxml.jackson.databind.JsonNode;

public record MCPTool(
    String name,
    String description,
    JsonNode inputSchema
) {
    public String getSchemaString() {
        return inputSchema != null ? inputSchema.toString() : "{}";
    }
}
