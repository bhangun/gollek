package tech.kayys.gollek.sdk.mcp;

import java.util.Map;

/**
 * Model representing a tool discovered from an MCP server.
 */
public record McpToolModel(
        String name,
        String description,
        Map<String, Object> inputSchema) {
}
