package tech.kayys.gollek.mcp.dto;

import java.util.Map;

/**
 * MCP Notification message (no response expected).
 */
public class MCPNotification extends JsonRpcMessage {
    private final String method;
    private final Map<String, Object> params;

    public MCPNotification(String method, Map<String, Object> params) {
        super("2.0", null); // Notifications have no ID
        this.method = method;
        this.params = params != null ? params : Map.of();
    }

    public String getMethod() {
        return method;
    }

    public Map<String, Object> getParams() {
        return params;
    }
}