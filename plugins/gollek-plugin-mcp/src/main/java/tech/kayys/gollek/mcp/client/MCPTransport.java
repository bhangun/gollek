package tech.kayys.gollek.mcp.client;

import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.mcp.dto.JsonRpcMessage;
import tech.kayys.gollek.mcp.dto.MCPRequest;
import tech.kayys.gollek.mcp.dto.MCPResponse;

import java.util.function.Consumer;

/**
 * Transport layer abstraction for MCP communication.
 * Supports stdio, HTTP, and WebSocket transports.
 */
public interface MCPTransport extends AutoCloseable {

    /**
     * Connect to the MCP server
     */
    Uni<Void> connect();

    /**
     * Send a request and wait for response
     */
    Uni<MCPResponse> sendRequest(MCPRequest request);

    /**
     * Send a notification (no response expected)
     */
    Uni<Void> sendNotification(String method, Object params);

    /**
     * Register handler for incoming messages
     */
    void onMessage(Consumer<JsonRpcMessage> handler);

    /**
     * Check if connected
     */
    boolean isConnected();

    /**
     * Disconnect from server
     */
    Uni<Void> disconnect();

    /**
     * Close and cleanup resources
     */
    @Override
    void close();
}