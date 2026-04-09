package tech.kayys.gollek.sdk.mcp;

public record McpAddRequest(
                String inlineJson,
                String filePath,
                String fromUrl,
                String fromRegistry,
                String name,
                String transport,
                String command,
                String url,
                String argsJson,
                String envJson,
                Boolean enabled) {
}
