package tech.kayys.gollek.provider.core.mcp;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Configuration for MCP client.
 */
public final class MCPClientConfig {

    public enum TransportType {
        STDIO, // Standard input/output (for local processes)
        HTTP, // HTTP/SSE transport
        WEBSOCKET // WebSocket transport
    }

    private final String name;
    private final TransportType transportType;
    private final String command; // For stdio
    private final String[] args; // For stdio
    private final String url; // For HTTP/WebSocket
    private final Map<String, String> headers; // For HTTP/WebSocket
    private final Duration timeout;
    private final Duration connectTimeout;
    private final boolean autoReconnect;
    private final int maxReconnectAttempts;
    private final Map<String, Object> metadata;

    private MCPClientConfig(Builder builder) {
        this.name = builder.name;
        this.transportType = builder.transportType;
        this.command = builder.command;
        this.args = builder.args;
        this.url = builder.url;
        this.headers = new HashMap<>(builder.headers);
        this.timeout = builder.timeout;
        this.connectTimeout = builder.connectTimeout;
        this.autoReconnect = builder.autoReconnect;
        this.maxReconnectAttempts = builder.maxReconnectAttempts;
        this.metadata = new HashMap<>(builder.metadata);
    }

    // Getters
    public String getName() {
        return name;
    }

    public TransportType getTransportType() {
        return transportType;
    }

    public String getCommand() {
        return command;
    }

    public String[] getArgs() {
        return args;
    }

    public String getUrl() {
        return url;
    }

    public Map<String, String> getHeaders() {
        return headers;
    }

    public Duration getTimeout() {
        return timeout;
    }

    public Duration getConnectTimeout() {
        return connectTimeout;
    }

    public boolean isAutoReconnect() {
        return autoReconnect;
    }

    public int getMaxReconnectAttempts() {
        return maxReconnectAttempts;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String name;
        private TransportType transportType = TransportType.STDIO;
        private String command;
        private String[] args = new String[0];
        private String url;
        private final Map<String, String> headers = new HashMap<>();
        private Duration timeout = Duration.ofSeconds(30);
        private Duration connectTimeout = Duration.ofSeconds(10);
        private boolean autoReconnect = true;
        private int maxReconnectAttempts = 3;
        private final Map<String, Object> metadata = new HashMap<>();

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder transportType(TransportType transportType) {
            this.transportType = transportType;
            return this;
        }

        public Builder command(String command) {
            this.command = command;
            return this;
        }

        public Builder args(String... args) {
            this.args = args;
            return this;
        }

        public Builder url(String url) {
            this.url = url;
            return this;
        }

        public Builder header(String key, String value) {
            this.headers.put(key, value);
            return this;
        }

        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        public Builder connectTimeout(Duration connectTimeout) {
            this.connectTimeout = connectTimeout;
            return this;
        }

        public Builder autoReconnect(boolean autoReconnect) {
            this.autoReconnect = autoReconnect;
            return this;
        }

        public Builder maxReconnectAttempts(int maxReconnectAttempts) {
            this.maxReconnectAttempts = maxReconnectAttempts;
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public MCPClientConfig build() {
            Objects.requireNonNull(name, "name is required");

            if (transportType == TransportType.STDIO) {
                Objects.requireNonNull(command, "command is required for stdio transport");
            } else {
                Objects.requireNonNull(url, "url is required for HTTP/WebSocket transport");
            }

            return new MCPClientConfig(this);
        }
    }

    @Override
    public String toString() {
        return "MCPClientConfig{" +
                "name='" + name + '\'' +
                ", transportType=" + transportType +
                ", command='" + command + '\'' +
                ", url='" + url + '\'' +
                '}';
    }
}
