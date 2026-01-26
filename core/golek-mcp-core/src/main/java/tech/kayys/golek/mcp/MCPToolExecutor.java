package tech.kayys.golek.provider.core.mcp;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Executes MCP tools with validation, error handling, and observability.
 */
@ApplicationScoped
public class MCPToolExecutor {

    private static final Logger LOG = Logger.getLogger(MCPToolExecutor.class);

    @Inject
    MCPClient mcpClient;

    @Inject
    MCPToolRegistry toolRegistry;

    /**
     * Execute a tool with arguments
     */
    public Uni<MCPToolResult> executeTool(
            String toolName,
            Map<String, Object> arguments) {
        return executeTool(toolName, arguments, Duration.ofSeconds(30));
    }

    /**
     * Execute tool with timeout
     */
    public Uni<MCPToolResult> executeTool(
            String toolName,
            Map<String, Object> arguments,
            Duration timeout) {
        LOG.debugf("Executing tool: %s with args: %s", toolName, arguments);

        // Validate tool exists
        Optional<MCPTool> toolOpt = toolRegistry.getTool(toolName);
        if (toolOpt.isEmpty()) {
            return Uni.createFrom().item(
                    MCPToolResult.error(toolName, "Tool not found: " + toolName));
        }

        MCPTool tool = toolOpt.get();

        // Validate arguments
        if (!tool.validateArguments(arguments)) {
            return Uni.createFrom().item(
                    MCPToolResult.error(toolName, "Invalid arguments for tool: " + toolName));
        }

        // Get connection
        Optional<String> connectionNameOpt = toolRegistry.getConnectionForTool(toolName);
        if (connectionNameOpt.isEmpty()) {
            return Uni.createFrom().item(
                    MCPToolResult.error(toolName, "No connection for tool: " + toolName));
        }

        String connectionName = connectionNameOpt.get();
        Optional<MCPConnection> connectionOpt = mcpClient.getConnection(connectionName);
        if (connectionOpt.isEmpty()) {
            return Uni.createFrom().item(
                    MCPToolResult.error(toolName, "Connection not found: " + connectionName));
        }

        MCPConnection connection = connectionOpt.get();

        // Execute with timeout
        return connection.callTool(toolName, arguments)
                .ifNoItem().after(timeout).fail()
                .onItem().transform(response -> convertResponse(toolName, response))
                .onFailure().recoverWithItem(error -> {
                    LOG.errorf(error, "Tool execution failed: %s", toolName);
                    return MCPToolResult.error(
                            toolName,
                            "Execution failed: " + error.getMessage());
                });
    }

    /**
     * Convert MCP response to tool result
     */
    private MCPToolResult convertResponse(String toolName, MCPResponse response) {
        if (!response.isSuccess()) {
            String errorMsg = response.getError() != null
                    ? response.getError().toString()
                    : "Unknown error";
            return MCPToolResult.error(toolName, errorMsg);
        }

        @SuppressWarnings("unchecked")
        Map<String, Object> result = (Map<String, Object>) response.getResult();
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> contentList = (List<Map<String, Object>>) result.get("content");

        if (contentList == null || contentList.isEmpty()) {
            return MCPToolResult.success(toolName, "");
        }

        var builder = MCPToolResult.builder()
                .toolName(toolName)
                .success(true);

        contentList.forEach(contentData -> {
            String type = (String) contentData.get("type");
            String text = (String) contentData.get("text");
            String data = (String) contentData.get("data");
            String mimeType = (String) contentData.get("mimeType");
            String uri = (String) contentData.get("uri");

            builder.addContent(
                    new MCPToolResult.Content(type, text, data, mimeType, uri));
        });

        return builder.build();
    }

    /**
     * Batch execute multiple tools
     */
    public Uni<Map<String, MCPToolResult>> executeTools(
            Map<String, Map<String, Object>> toolCalls) {
        return Uni.combine().all().unis(
                toolCalls.entrySet().stream()
                        .map(entry -> executeTool(entry.getKey(), entry.getValue())
                                .onItem().transform(result -> Map.entry(entry.getKey(), result)))
                        .toList())
                .with(results -> {
                    return results.stream()
                            .collect(java.util.stream.Collectors.toMap(
                                    Map.Entry::getKey,
                                    Map.Entry::getValue));
                });
    }
}