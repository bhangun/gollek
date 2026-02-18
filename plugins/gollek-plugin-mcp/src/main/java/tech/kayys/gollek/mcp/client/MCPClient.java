package tech.kayys.gollek.mcp.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.mcp.dto.MCPConnection;

import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Production-ready MCP client.
 * Manages connection to MCP server and exposes tools/resources/prompts.
 */
@ApplicationScoped
public class MCPClient implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(MCPClient.class);

    @Inject
    ObjectMapper objectMapper;

    private final Map<String, MCPConnection> connections = new ConcurrentHashMap<>();
    private final AtomicLong requestIdCounter = new AtomicLong(0);

    /**
     * Connect to an MCP server
     */
    public Uni<MCPConnection> connect(MCPClientConfig config) {
        if (connections.containsKey(config.getName())) {
            return Uni.createFrom().failure(
                    new IllegalStateException("Already connected to: " + config.getName()));
        }

        MCPTransport transport = createTransport(config);
        MCPConnection connection = new MCPConnection(config, transport, objectMapper);

        return connection.connect()
                .onItem().invoke(() -> {
                    connections.put(config.getName(), connection);
                    LOG.infof("Connected to MCP server: %s", config.getName());
                })
                .replaceWith(connection);
    }

    private MCPTransport createTransport(MCPClientConfig config) {
        return switch (config.getTransportType()) {
            case STDIO -> new StdioTransport(config, objectMapper);
            case HTTP -> throw new UnsupportedOperationException("HTTP transport not yet implemented");
            case WEBSOCKET -> throw new UnsupportedOperationException("WebSocket transport not yet implemented");
        };
    }

    /**
     * Get connection by name
     */
    public Optional<MCPConnection> getConnection(String name) {
        return Optional.ofNullable(connections.get(name));
    }

    /**
     * Disconnect from server
     */
    public Uni<Void> disconnect(String name) {
        MCPConnection connection = connections.remove(name);
        if (connection != null) {
            return connection.disconnect()
                    .onItem().invoke(() -> LOG.infof("Disconnected from MCP server: %s", name));
        }
        return Uni.createFrom().voidItem();
    }

    /**
     * Get all active connections
     */
    public List<MCPConnection> getActiveConnections() {
        return new ArrayList<>(connections.values());
    }

    @Override
    public void close() {
        LOG.info("Closing all MCP connections...");
        connections.values().forEach(MCPConnection::close);
        connections.clear();
    }

    /**
     * Generate unique request ID
     */
    public long generateRequestId() {
        return requestIdCounter.incrementAndGet();
    }
}
