package tech.kayys.golek.provider.core.mcp;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

/**
 * Base JSON-RPC 2.0 message.
 * MCP uses JSON-RPC 2.0 for all communication.
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.EXISTING_PROPERTY, property = "jsonrpc", visible = true)
@JsonSubTypes({
        @JsonSubTypes.Type(value = MCPRequest.class, name = "2.0"),
        @JsonSubTypes.Type(value = MCPResponse.class, name = "2.0"),
        @JsonSubTypes.Type(value = MCPNotification.class, name = "2.0")
})
public abstract class JsonRpcMessage {

    private static final String VERSION = "2.0";

    private final String jsonrpc;

    protected JsonRpcMessage() {
        this.jsonrpc = VERSION;
    }

    @JsonCreator
    protected JsonRpcMessage(@JsonProperty("jsonrpc") String jsonrpc) {
        this.jsonrpc = jsonrpc != null ? jsonrpc : VERSION;
    }

    public String getJsonrpc() {
        return jsonrpc;
    }

    public abstract boolean isRequest();

    public abstract boolean isResponse();

    public abstract boolean isNotification();
}
