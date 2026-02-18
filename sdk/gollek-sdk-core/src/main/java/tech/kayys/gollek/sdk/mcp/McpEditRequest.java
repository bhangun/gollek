package tech.kayys.gollek.sdk.core.mcp;

public record McpEditRequest(
        String name,
        String transport,
        String command,
        String url,
        String argsJson,
        boolean clearArgs,
        String envJson,
        boolean clearEnv,
        Boolean enabled) {
}

