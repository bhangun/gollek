package tech.kayys.gollek.sdk.core.mcp;

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
