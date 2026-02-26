package tech.kayys.gollek.sdk.mcp;

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
