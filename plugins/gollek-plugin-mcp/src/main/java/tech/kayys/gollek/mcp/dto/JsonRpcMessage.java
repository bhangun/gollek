package tech.kayys.gollek.mcp.dto;

import java.util.Map;

/**
 * Base class for JSON-RPC messages in MCP protocol.
 */
public abstract class JsonRpcMessage {
    protected final String jsonrpc;
    protected final Object id;

    public JsonRpcMessage(String jsonrpc, Object id) {
        this.jsonrpc = jsonrpc;
        this.id = id;
    }

    public String getJsonrpc() {
        return jsonrpc;
    }

    public Object getId() {
        return id;
    }

    public boolean isResponse() {
        return this instanceof MCPResponse;
    }
}