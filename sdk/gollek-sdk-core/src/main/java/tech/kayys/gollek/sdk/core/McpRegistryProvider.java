package tech.kayys.gollek.sdk.core;

import tech.kayys.gollek.sdk.mcp.McpRegistryManager;

/**
 * Marker interface for SDK implementations that support MCP registry
 * management.
 *
 * <p>
 * Implementations of {@link GollekSdk} that also implement this interface
 * can provide access to the {@link McpRegistryManager} for managing MCP
 * servers.
 */
public interface McpRegistryProvider {

    /**
     * Returns the MCP Registry Manager for managing MCP servers.
     *
     * @return the MCP registry manager
     * @throws IllegalStateException if MCP is not available
     */
    McpRegistryManager mcpRegistry();
}
