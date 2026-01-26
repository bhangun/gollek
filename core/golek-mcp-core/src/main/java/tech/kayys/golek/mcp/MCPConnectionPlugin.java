package tech.kayys.golek.provider.core.mcp;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.golek.provider.core.plugin.ConfigurablePlugin;
import tech.kayys.golek.provider.core.plugin.Plugin.PluginMetadata;

import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Plugin that manages MCP server connections.
 * This is a LIFECYCLE plugin (not phase-bound).
 * It initializes MCP connections and makes them available to other plugins.
 */
@ApplicationScoped
public class MCPConnectionPlugin implements ConfigurablePlugin {

    private static final Logger LOG = Logger.getLogger(MCPConnectionPlugin.class);
    private static final String PLUGIN_ID = "mcp-connections";

    @Inject
    MCPClient mcpClient;

    private final Map<String, MCPConnection> connections = new ConcurrentHashMap<>();
    private Map<String, Object> config = new HashMap<>();
    private boolean initialized = false;

    @Override
    public String id() {
        return PLUGIN_ID;
    }

    @Override
    public String name() {
        return "MCP Connection Manager";
    }

    @Override
    public String version() {
        return "1.0.0";
    }

    @Override
    public Uni<Void> initialize(PluginContext context) {
        LOG.info("Initializing MCP connections...");

        this.config = new HashMap<>(context.config());

        // Get MCP server configurations
        @SuppressWarnings("unchecked")
        Map<String, Map<String, Object>> servers = (Map<String, Map<String, Object>>) config.get("servers");

        if (servers == null || servers.isEmpty()) {
            LOG.warn("No MCP servers configured");
            initialized = true;
            return Uni.createFrom().voidItem();
        }

        // Connect to all servers
        List<Uni<MCPConnection>> connectionUnis = servers.entrySet().stream()
                .map(entry -> {
                    String serverName = entry.getKey();
                    Map<String, Object> serverConfig = entry.getValue();

                    MCPClientConfig clientConfig = buildClientConfig(serverName, serverConfig);

                    return mcpClient.connect(clientConfig)
                            .onItem().invoke(connection -> {
                                connections.put(serverName, connection);

                                // Share connection with other plugins
                                context.putSharedData("mcp.connection." + serverName, connection);

                                LOG.infof("Connected to MCP server: %s", serverName);
                            })
                            .onFailure()
                            .invoke(error -> LOG.errorf(error, "Failed to connect to MCP server: %s", serverName));
                })
                .toList();

        return Uni.join().all(connectionUnis).andCollectFailures()
                .replaceWithVoid()
                .onItem().invoke(() -> {
                    initialized = true;
                    LOG.infof("Initialized %d MCP connections", connections.size());
                });
    }

    @Override
    public Uni<Void> shutdown() {
        LOG.info("Shutting down MCP connections...");

        List<Uni<Void>> closeUnis = connections.values().stream()
                .map(MCPConnection::close)
                .toList();

        return Uni.join().all(closeUnis).andCollectFailures()
                .replaceWithVoid()
                .onItem().invoke(() -> {
                    connections.clear();
                    initialized = false;
                    LOG.info("All MCP connections closed");
                });
    }

    @Override
    public PluginHealth health() {
        if (!initialized) {
            return PluginHealth.unhealthy("Not initialized");
        }

        long activeConnections = connections.values().stream()
                .filter(MCPConnection::isConnected)
                .count();

        if (activeConnections == 0 && !connections.isEmpty()) {
            return PluginHealth.unhealthy(
                    "No active MCP connections",
                    Map.of("total", connections.size(), "active", 0));
        }

        if (activeConnections < connections.size()) {
            return PluginHealth.degraded(
                    String.format("%d/%d MCP connections active", activeConnections, connections.size()),
                    Map.of("total", connections.size(), "active", activeConnections));
        }

        return PluginHealth.healthy(
                String.format("%d MCP connections active", activeConnections));
    }

    @Override
    public Uni<Void> onConfigUpdate(Map<String, Object> newConfig) {
        LOG.info("MCP connection configuration updated - restart required");
        // Connection changes require restart
        return Uni.createFrom().voidItem();
    }

    @Override
    public Map<String, Object> currentConfig() {
        return new HashMap<>(config);
    }

    /**
     * Get connection by server name
     */
    public Optional<MCPConnection> getConnection(String serverName) {
        return Optional.ofNullable(connections.get(serverName));
    }

    /**
     * Get all active connections
     */
    public List<MCPConnection> getAllConnections() {
        return new ArrayList<>(connections.values());
    }

    private MCPClientConfig buildClientConfig(String name, Map<String, Object> config) {
        String transport = (String) config.getOrDefault("transport", "stdio");

        var builder = MCPClientConfig.builder()
                .name(name)
                .transportType(MCPClientConfig.TransportType.valueOf(transport.toUpperCase()));

        if ("stdio".equalsIgnoreCase(transport)) {
            builder.command((String) config.get("command"))
                    .args((List<String>) config.getOrDefault("args", List.of()))
                    .env((Map<String, String>) config.getOrDefault("env", Map.of()));
        } else if ("http".equalsIgnoreCase(transport) || "websocket".equalsIgnoreCase(transport)) {
            builder.url((String) config.get("url"));
        }

        return builder.build();
    }

    @Override
    public PluginMetadata metadata() {
        return PluginMetadata.builder()
                .id(id())
                .name(name())
                .version(version())
                .description("Manages MCP server connections and lifecycle")
                .author("Kayys Tech")
                .tag("mcp")
                .tag("infrastructure")
                .tag("lifecycle")
                .property("type", "connection-manager")
                .build();
    }
}