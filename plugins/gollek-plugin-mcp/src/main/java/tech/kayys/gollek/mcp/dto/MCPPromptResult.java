package tech.kayys.gollek.mcp.dto;

import java.util.List;
import java.util.Map;

/**
 * Result of an MCP prompt execution.
 */
public class MCPPromptResult {
    private final String promptName;
    private final String description;
    private final List<MCPPromptMessage> messages;
    private final Map<String, Object> metadata;

    private MCPPromptResult(Builder builder) {
        this.promptName = builder.promptName;
        this.description = builder.description;
        this.messages = builder.messages;
        this.metadata = builder.metadata != null ? builder.metadata : Map.of();
    }

    public String getPromptName() {
        return promptName;
    }

    public String getDescription() {
        return description;
    }

    public List<MCPPromptMessage> getMessages() {
        return messages;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String promptName;
        private String description;
        private List<MCPPromptMessage> messages;
        private Map<String, Object> metadata;

        public Builder promptName(String promptName) {
            this.promptName = promptName;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder messages(List<MCPPromptMessage> messages) {
            this.messages = messages;
            return this;
        }

        public Builder message(MCPPromptMessage message) {
            this.messages.add(message);
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public MCPPromptResult build() {
            return new MCPPromptResult(this);
        }
    }
}