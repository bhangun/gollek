package tech.kayys.wayang.tool.mcp;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.wayang.tool.spi.Tool;

import java.util.Collections;
import java.util.Map;

/**
 * MCP Tool implementation.
 * Wraps an MCP tool execution.
 */
public class McpTool implements Tool {

    private final String id;
    private final String name;
    private final String description;

    public McpTool(String id, String name, String description) {
        this.id = id;
        this.name = name;
        this.description = description;
    }

    @Override
    public String id() {
        return id;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public String description() {
        return description;
    }

    @Override
    public Map<String, Object> inputSchema() {
        return Collections.emptyMap(); // Placeholder
    }

    @Override
    public Uni<Map<String, Object>> execute(Map<String, Object> arguments, Map<String, Object> context) {
        // TODO: Implement actual MCP call
        return Uni.createFrom().item(Map.of("status", "executed", "tool", "mcp"));
    }
}
