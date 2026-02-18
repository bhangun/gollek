package tech.kayys.gollek.mcp.dto;

import java.util.Map;

/**
 * Result of an MCP tool execution.
 */
public class MCPToolResult {

    private final String toolName;
    private final boolean success;
    private final String errorMessage;
    private final Map<String, Object> output;

    private MCPToolResult(Builder builder) {
        this.toolName = builder.toolName;
        this.success = builder.success;
        this.errorMessage = builder.errorMessage;
        this.output = builder.output;
    }

    public String getToolName() {
        return toolName;
    }

    public boolean isSuccess() {
        return success;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public Map<String, Object> getOutput() {
        return output;
    }

    public String getAllText() {
        if (output != null && output.containsKey("content")) {
            return output.get("content").toString();
        }
        return output != null ? output.toString() : "";
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String toolName;
        private boolean success = true;
        private String errorMessage;
        private Map<String, Object> output;

        public Builder toolName(String toolName) {
            this.toolName = toolName;
            return this;
        }

        public Builder success(boolean success) {
            this.success = success;
            return this;
        }

        public Builder errorMessage(String errorMessage) {
            this.errorMessage = errorMessage;
            return this;
        }

        public Builder output(Map<String, Object> output) {
            this.output = output;
            return this;
        }

        public MCPToolResult build() {
            return new MCPToolResult(this);
        }
    }
}