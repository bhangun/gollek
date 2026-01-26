package tech.kayys.golek.provider.core.mcp;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.HashMap;
import java.util.Map;

/**
 * MCP JSON-RPC error object.
 */
public final class MCPError {

    // Standard JSON-RPC error codes
    public static final int PARSE_ERROR = -32700;
    public static final int INVALID_REQUEST = -32600;
    public static final int METHOD_NOT_FOUND = -32601;
    public static final int INVALID_PARAMS = -32602;
    public static final int INTERNAL_ERROR = -32603;

    // MCP-specific error codes
    public static final int RESOURCE_NOT_FOUND = -32001;
    public static final int TOOL_NOT_FOUND = -32002;
    public static final int TOOL_EXECUTION_ERROR = -32003;
    public static final int PROMPT_NOT_FOUND = -32004;
    public static final int UNAUTHORIZED = -32005;

    private final int code;
    private final String message;
    private final Map<String, Object> data;

    @JsonCreator
    public MCPError(
            @JsonProperty("code") int code,
            @JsonProperty("message") String message,
            @JsonProperty("data") Map<String, Object> data) {
        this.code = code;
        this.message = message;
        this.data = data != null ? new HashMap<>(data) : new HashMap<>();
    }

    public int getCode() {
        return code;
    }

    public String getMessage() {
        return message;
    }

    public Map<String, Object> getData() {
        return data;
    }

    public static MCPError parseError(String details) {
        return new MCPError(PARSE_ERROR, "Parse error", Map.of("details", details));
    }

    public static MCPError invalidRequest(String details) {
        return new MCPError(INVALID_REQUEST, "Invalid request", Map.of("details", details));
    }

    public static MCPError methodNotFound(String method) {
        return new MCPError(METHOD_NOT_FOUND, "Method not found", Map.of("method", method));
    }

    public static MCPError invalidParams(String details) {
        return new MCPError(INVALID_PARAMS, "Invalid params", Map.of("details", details));
    }

    public static MCPError internalError(String details) {
        return new MCPError(INTERNAL_ERROR, "Internal error", Map.of("details", details));
    }

    public static MCPError resourceNotFound(String resourceId) {
        return new MCPError(RESOURCE_NOT_FOUND, "Resource not found", Map.of("resourceId", resourceId));
    }

    public static MCPError toolNotFound(String toolName) {
        return new MCPError(TOOL_NOT_FOUND, "Tool not found", Map.of("toolName", toolName));
    }

    public static MCPError toolExecutionError(String toolName, String error) {
        return new MCPError(TOOL_EXECUTION_ERROR, "Tool execution error",
                Map.of("toolName", toolName, "error", error));
    }

    @Override
    public String toString() {
        return "MCPError{" +
                "code=" + code +
                ", message='" + message + '\'' +
                '}';
    }
}