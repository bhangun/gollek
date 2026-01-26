package tech.kayys.golek.provider.core.mcp;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * MCP JSON-RPC notification message.
 * Notifications do not expect a response.
 */
public final class MCPNotification extends JsonRpcMessage {

    @NotBlank
    private final String method;

    private final Map<String, Object> params;

    @JsonCreator
    public MCPNotification(
            @JsonProperty("method") String method,
            @JsonProperty("params") Map<String, Object> params) {
        super();
        this.method = Objects.requireNonNull(method, "method is required");
        this.params = params != null ? new HashMap<>(params) : new HashMap<>();
    }

    public String getMethod() {
        return method;
    }

    public Map<String, Object> getParams() {
        return params;
    }

    @Override
    public boolean isRequest() {
        return false;
    }

    @Override
    public boolean isResponse() {
        return false;
    }

    @Override
    public boolean isNotification() {
        return true;
    }

    public static MCPNotification create(String method, Map<String, Object> params) {
        return new MCPNotification(method, params);
    }

    @Override
    public String toString() {
        return "MCPNotification{" +
                "method='" + method + '\'' +
                '}';
    }
}