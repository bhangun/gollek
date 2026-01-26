package tech.kayys.golek.provider.core.mcp;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Central registry for all MCP tools across connections.
 * Thread-safe and supports multi-tenant tool discovery.
 */
@ApplicationScoped
public class MCPToolRegistry {

    private static final Logger LOG = Logger.getLogger(MCPToolRegistry.class);

    // connectionName -> tools
    private final Map<String, Map<String, MCPTool>> toolsByConnection = new ConcurrentHashMap<>();

    // toolName -> connectionName (for quick lookup)
    private final Map<String, String> toolToConnection = new ConcurrentHashMap<>();

    /**
     * Register tools from a connection
     */
    public void registerConnection(MCPConnection connection) {
        String connectionName = connection.getConfig().getName();
        Map<String, MCPTool> tools = connection.getTools();

        toolsByConnection.put(connectionName, new ConcurrentHashMap<>(tools));

        // Update quick lookup map
        tools.keySet().forEach(toolName -> toolToConnection.put(toolName, connectionName));

        LOG.infof("Registered %d tools from connection: %s", tools.size(), connectionName);
    }

    /**
     * Unregister connection
     */
    public void unregisterConnection(String connectionName) {
        Map<String, MCPTool> tools = toolsByConnection.remove(connectionName);
        if (tools != null) {
            tools.keySet().forEach(toolToConnection::remove);
            LOG.infof("Unregistered connection: %s", connectionName);
        }
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
     * Get connection for tool
     */
    public Optional<String> getConnectionForTool(String toolName) {
        return Optional.ofNullable(toolToConnection.get(toolName));
    }

    /**
     * Get all tools
     */
    public List<MCPTool> getAllTools() {
        return toolsByConnection.values().stream()
                .flatMap(tools -> tools.values().stream())
                .toList();
    }

    /**
     * Get tools by connection
     */
    public List<MCPTool> getToolsByConnection(String connectionName) {
        Map<String, MCPTool> tools = toolsByConnection.get(connectionName);
        return tools != null ? new ArrayList<>(tools.values()) : Collections.emptyList();
    }

    /**
     * Search tools by keyword
     */
    public List<MCPTool> searchTools(String keyword) {
        String lowerKeyword = keyword.toLowerCase();
        return getAllTools().stream()
                .filter(tool -> tool.getName().toLowerCase().contains(lowerKeyword) ||
                        (tool.getDescription() != null &&
                                tool.getDescription().toLowerCase().contains(lowerKeyword)))
                .toList();
    }

    /**
     * Get tool count
     */
    public int getTotalToolCount() {
        return toolToConnection.size();
    }

    /**
     * Get connection count
     */
    public int getConnectionCount() {
        return toolsByConnection.size();
    }

    /**
     * Check if tool exists
     */
    public boolean hasTool(String toolName) {
        return toolToConnection.containsKey(toolName);
    }

    /**
     * Clear all registrations
     */
    public void clear() {
        toolsByConnection.clear();
        toolToConnection.clear();
        LOG.info("Cleared all tool registrations");
    }
}