package tech.kayys.gollek.mcp.tool;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.mcp.client.MCPClient;
import tech.kayys.gollek.mcp.dto.MCPConnection;
import tech.kayys.gollek.mcp.dto.MCPResponse;
import tech.kayys.gollek.mcp.dto.MCPToolResult;
import tech.kayys.gollek.mcp.exception.ConnectionNotFoundException;

import org.jboss.logging.Logger;

import java.util.Map;
import java.util.Optional;

/**
 * Executes tools via MCP connections.
 */
@ApplicationScoped
public class MCPToolExecutor {

    private static final Logger LOG = Logger.getLogger(MCPToolExecutor.class);

    @Inject
    MCPToolRegistry toolRegistry;

    @Inject
    MCPClient mcpClient;

    /**
     * Execute multiple tools in parallel
     */
    public Uni<Map<String, MCPToolResult>> executeTools(Map<String, Map<String, Object>> toolCalls) {
        LOG.debugf("Executing %d tools", toolCalls.size());
        if (toolCalls.isEmpty()) {
            return Uni.createFrom().item(Map.of());
        }

        return Uni.combine().all().unis(
                toolCalls.entrySet().stream()
                        .map(entry -> executeTool(entry.getKey(), entry.getValue()))
                        .toList())
                .with(results -> results.stream()
                            .map(MCPToolResult.class::cast)
                            .collect(java.util.stream.Collectors.toMap(
                                    MCPToolResult::getToolName,
                                    result -> result
                            )));
    }

    /**
     * Execute a single tool
     */
    public Uni<MCPToolResult> executeTool(String toolName, Map<String, Object> arguments) {
        LOG.debugf("Executing tool: %s", toolName);

        // Find which connection provides this tool
        Optional<String> connectionNameOpt = toolRegistry.getConnectionForTool(toolName);
        if (connectionNameOpt.isEmpty()) {
            return Uni.createFrom().failure(
                    new IllegalArgumentException("Tool not found: " + toolName));
        }

        String connectionName = connectionNameOpt.get();
        Optional<MCPConnection> connectionOpt = mcpClient.getConnection(connectionName);
        if (connectionOpt.isEmpty()) {
            return Uni.createFrom().failure(
                    new ConnectionNotFoundException("Connection not found: " + connectionName));
        }

        MCPConnection connection = connectionOpt.get();

        // Execute the tool
        return connection.callTool(toolName, arguments)
                .onItem().transform(response -> convertResponse(toolName, response))
                .onFailure().recoverWithItem(error -> {
                    LOG.warnf("Tool %s execution failed: %s", toolName, error.getMessage());
                    return MCPToolResult.builder()
                            .toolName(toolName)
                            .success(false)
                            .errorMessage(error.getMessage())
                            .build();
                });
    }

    /**
     * Convert MCP response to tool result
     */
    private MCPToolResult convertResponse(String toolName, MCPResponse response) {
        if (!response.isSuccess()) {
            return MCPToolResult.builder()
                    .toolName(toolName)
                    .success(false)
                    .errorMessage(response.getError().getMessage())
                    .build();
        }

        @SuppressWarnings("unchecked")
        Map<String, Object> result = (Map<String, Object>) response.getResult();

        return MCPToolResult.builder()
                .toolName(toolName)
                .success(true)
                .output(result)
                .build();
    }
}
