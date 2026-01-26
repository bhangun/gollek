package tech.kayys.golek.provider.core.mcp;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * MCP JSON-RPC request message.
 */
public final class MCPRequest extends JsonRpcMessage {

    @NotNull
    private final Object id;

    @NotBlank
    private final String method;

    private final Map<String, Object> params;

    @JsonCreator
    public MCPRequest(
            @JsonProperty("id") Object id,
            @JsonProperty("method") String method,
            @JsonProperty("params") Map<String, Object> params) {
        super();
        this.id = Objects.requireNonNull(id, "id is required");
        this.method = Objects.requireNonNull(method, "method is required");
        this.params = params != null ? new HashMap<>(params) : new HashMap<>();
    }

    public Object getId() {
        return id;
    }

    public String getMethod() {
        return method;
    }

    public Map<String, Object> getParams() {
        return params;
    }

    @SuppressWarnings("unchecked")
    public <T> T getParam(String key, Class<T> type) {
        Object value = params.get(key);
        if (value != null && type.isInstance(value)) {
            return (T) value;
        }
        return null;
    }

    @Override
    public boolean isRequest() {
        return true;
    }

    @Override
    public boolean isResponse() {
        return false;
    }

    @Override
    public boolean isNotification() {
        return false;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private Object id;
        private String method;
        private final Map<String, Object> params = new HashMap<>();

        public Builder id(Object id) {
            this.id = id;
            return this;
        }

        public Builder method(String method) {
            this.method = method;
            return this;
        }

        public Builder param(String key, Object value) {
            this.params.put(key, value);
            return this;
        }

        public Builder params(Map<String, Object> params) {
            this.params.putAll(params);
            return this;
        }

        public MCPRequest build() {
            return new MCPRequest(id, method, params);
        }
    }

    @Override
    public String toString() {
        return "MCPRequest{" +
                "id=" + id +
                ", method='" + method + '\'' +
                ", params=" + params.keySet() +
                '}';
    }
}