package tech.kayys.gollek.mcp.dto;

import java.util.Map;

/**
 * MCP Response message.
 */
public class MCPResponse extends JsonRpcMessage {
    private final Object result;
    private final MCPError error;

    public MCPResponse(Object id, Object result, MCPError error) {
        super("2.0", id);
        this.result = result;
        this.error = error;
    }

    public Object getResult() {
        return result;
    }

    public MCPError getError() {
        return error;
    }

    public boolean isSuccess() {
        return error == null;
    }

    public static MCPResponse success(Object id, Object result) {
        return new MCPResponse(id, result, null);
    }

    public static MCPResponse error(Object id, MCPError error) {
        return new MCPResponse(id, null, error);
    }
}