package tech.kayys.golek.provider.core.mcp;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import io.smallrye.config.WithName;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Configuration for MCP provider.
 * Supports multiple MCP server connections.
 */
@ConfigMapping(prefix = "wayang.inference.mcp")
public interface MCPProviderConfiguration {

    /**
     * Enable MCP provider
     */
    @WithDefault("false")
    boolean enabled();

    /**
     * MCP servers to connect to
     */
    @WithName("servers")
    Map<String, MCPServerConfig> servers();

    /**
     * Default timeout for MCP operations
     */
    @WithDefault("30s")
    String defaultTimeout();

    /**
     * Enable tool execution
     */
    @WithDefault("true")
    boolean toolsEnabled();

    /**
     * Enable resource access
     */
    @WithDefault("true")
    boolean resourcesEnabled();

    /**
     * Enable prompt execution
     */
    @WithDefault("true")
    boolean promptsEnabled();

    /**
     * Resource cache configuration
     */
    @WithName("cache")
    CacheConfig cache();

    interface MCPServerConfig {
        /**
         * Display name for the server
         */
        String name();

        /**
         * Transport type (stdio, http, websocket)
         */
        @WithDefault("stdio")
        String transport();

        /**
         * Command to start stdio server
         */
        Optional<String> command();

        /**
         * Arguments for stdio command
         */
        @WithDefault("")
        List<String> args();

        /**
         * Environment variables for stdio server
         */
        @WithDefault("")
        Map<String, String> env();

        /**
         * HTTP/WebSocket URL
         */
        Optional<String> url();

        /**
         * Auto-reconnect on failure
         */
        @WithDefault("true")
        boolean autoReconnect();

        /**
         * Max reconnection attempts
         */
        @WithDefault("3")
        int maxReconnectAttempts();
    }

    interface CacheConfig {
        /**
         * Enable resource caching
         */
        @WithDefault("true")
        boolean enabled();

        /**
         * Maximum cache size
         */
        @WithDefault("1000")
        int maxSize();

        /**
         * Cache TTL
         */
        @WithDefault("15m")
        String ttl();
    }
}