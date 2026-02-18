package tech.kayys.gollek.mcp.dto;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * MCP Request message.
 */
public class MCPRequest extends JsonRpcMessage {
    private final String method;
    private final Map<String, Object> params;

    private MCPRequest(Builder builder) {
        super("2.0", builder.id);
        this.method = builder.method;
        this.params = new HashMap<>(builder.params);
    }

    public String getMethod() {
        return method;
    }

    public Map<String, Object> getParams() {
        return params;
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

        public Builder params(Map<String, Object> params) {
            this.params.putAll(params);
            return this;
        }

        public Builder param(String key, Object value) {
            this.params.put(key, value);
            return this;
        }

        public MCPRequest build() {
            Objects.requireNonNull(method, "method is required");
            return new MCPRequest(this);
        }
    }
}