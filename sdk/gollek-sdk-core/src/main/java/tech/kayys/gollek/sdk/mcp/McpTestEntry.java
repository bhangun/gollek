package tech.kayys.gollek.sdk.core.mcp;

public record McpTestEntry(
        String name,
        boolean success,
        int tools,
        int resources,
        int prompts,
        String error) {
}

