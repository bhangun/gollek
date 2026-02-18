package tech.kayys.gollek.mcp.dto;

/**
 * MCP Error representation.
 */
public class MCPError {
    private final int code;
    private final String message;
    private final Object data;

    public MCPError(int code, String message, Object data) {
        this.code = code;
        this.message = message;
        this.data = data;
    }

    public int getCode() {
        return code;
    }

    public String getMessage() {
        return message;
    }

    public Object getData() {
        return data;
    }

    @Override
    public String toString() {
        return "MCPError{" +
                "code=" + code +
                ", message='" + message + '\'' +
                ", data=" + data +
                '}';
    }
}