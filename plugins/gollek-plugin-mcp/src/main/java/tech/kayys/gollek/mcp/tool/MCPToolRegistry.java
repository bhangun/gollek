package tech.kayys.gollek.mcp.tool;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import tech.kayys.gollek.mcp.client.MCPClient;
import tech.kayys.gollek.mcp.dto.MCPConnection;
import tech.kayys.gollek.mcp.dto.MCPTool;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry for MCP tools across all connections.
 */
@ApplicationScoped
public class MCPToolRegistry {

    private static final Logger LOG = Logger.getLogger(MCPToolRegistry.class);

    // toolName -> connectionName
    private final Map<String, String> toolToConnection = new ConcurrentHashMap<>();

    // connectionName -> tools
    private final Map<String, Map<String, MCPTool>> toolsByConnection = new ConcurrentHashMap<>();

    /**
     * Register tools from a connection
     */
    public void registerConnection(MCPConnection connection) {
        String connectionName = connection.getConfig().getName();
        Map<String, MCPTool> tools = connection.getTools();

        toolsByConnection.put(connectionName, new ConcurrentHashMap<>(tools));

        tools.keySet().forEach(toolName -> toolToConnection.put(toolName, connectionName));

        LOG.infof("Registered %d tools from connection: %s", tools.size(), connectionName);
    }

    /**
     * Get connection name for a tool
     */
    public Optional<String> getConnectionForTool(String toolName) {
        return Optional.ofNullable(toolToConnection.get(toolName));
    }

    /**
     * Get tool by name
     */
    public Optional<MCPTool> getTool(String toolName) {
        String connectionName = toolToConnection.get(toolName);
        if (connectionName == null) {
            return Optional.empty();
        }

        Map<String, MCPTool> tools = toolsByConnection.get(connectionName);
        return Optional.ofNullable(tools != null ? tools.get(toolName) : null);
    }

    /**
     * Get all tools
     */
    public Map<String, MCPTool> getAllTools() {
        Map<String, MCPTool> allTools = new ConcurrentHashMap<>();
        toolsByConnection.values().forEach(allTools::putAll);
        return allTools;
    }
}