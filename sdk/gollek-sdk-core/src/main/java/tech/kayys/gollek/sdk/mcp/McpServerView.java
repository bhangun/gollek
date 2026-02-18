package tech.kayys.gollek.sdk.core.mcp;

public record McpServerView(
        String name,
        boolean enabled,
        String transport,
        String command,
        int argsCount,
        int envKeys,
        String url,
        String rawJson) {
}

