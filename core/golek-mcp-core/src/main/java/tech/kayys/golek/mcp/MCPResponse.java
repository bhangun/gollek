package tech.kayys.golek.provider.core.mcp;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotNull;

import java.util.Objects;

/**
 * MCP JSON-RPC response message.
 */
public final class MCPResponse extends JsonRpcMessage {

    @NotNull
    private final Object id;

    private final Object result;
    private final MCPError error;

    @JsonCreator
    public MCPResponse(
            @JsonProperty("id") Object id,
            @JsonProperty("result") Object result,
            @JsonProperty("error") MCPError error) {
        super();
        this.id = Objects.requireNonNull(id, "id is required");
        this.result = result;
        this.error = error;

        if (result == null && error == null) {
            throw new IllegalArgumentException("Either result or error must be present");
        }
        if (result != null && error != null) {
            throw new IllegalArgumentException("Cannot have both result and error");
        }
    }

    public Object getId() {
        return id;
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

    public boolean isError() {
        return error != null;
    }

    @SuppressWarnings("unchecked")
    public <T> T getResultAs(Class<T> type) {
        if (result != null && type.isInstance(result)) {
            return (T) result;
        }
        return null;
    }

    @Override
    public boolean isRequest() {
        return false;
    }

    @Override
    public boolean isResponse() {
        return true;
    }

    @Override
    public boolean isNotification() {
        return false;
    }

    public static MCPResponse success(Object id, Object result) {
        return new MCPResponse(id, result, null);
    }

    public static MCPResponse error(Object id, MCPError error) {
        return new MCPResponse(id, null, error);
    }

    @Override
    public String toString() {
        return "MCPResponse{" +
                "id=" + id +
                ", success=" + isSuccess() +
                (error != null ? ", error=" + error : "") +
                '}';
    }
}
