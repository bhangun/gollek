package tech.kayys.golek.provider.core.mcp;

import java.time.Instant;
import java.util.*;

/**
 * Result of executing an MCP prompt.
 */
public final class MCPPromptResult {

    private final String promptName;
    private final List<MCPPromptMessage> messages;
    private final String description;
    private final Map<String, Object> metadata;
    private final Instant timestamp;

    public MCPPromptResult(
            String promptName,
            List<MCPPromptMessage> messages,
            String description,
            Map<String, Object> metadata,
            Instant timestamp) {
        this.promptName = Objects.requireNonNull(promptName, "promptName");
        this.messages = messages != null
                ? Collections.unmodifiableList(new ArrayList<>(messages))
                : Collections.emptyList();
        this.description = description;
        this.metadata = metadata != null
                ? Collections.unmodifiableMap(new HashMap<>(metadata))
                : Collections.emptyMap();
        this.timestamp = timestamp != null ? timestamp : Instant.now();
    }

    // Getters
    public String getPromptName() {
        return promptName;
    }

    public List<MCPPromptMessage> getMessages() {
        return messages;
    }

    public String getDescription() {
        return description;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public Instant getTimestamp() {
        return timestamp;
    }

    /**
     * Get combined text from all messages
     */
    public String getCombinedText() {
        return messages.stream()
                .map(MCPPromptMessage::getAllText)
                .filter(text -> !text.isEmpty())
                .reduce("", (a, b) -> a + "\n\n" + b)
                .trim();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String promptName;
        private final List<MCPPromptMessage> messages = new ArrayList<>();
        private String description;
        private final Map<String, Object> metadata = new HashMap<>();
        private Instant timestamp = Instant.now();

        public Builder promptName(String promptName) {
            this.promptName = promptName;
            return this;
        }

        public Builder message(MCPPromptMessage message) {
            this.messages.add(message);
            return this;
        }

        public Builder messages(List<MCPPromptMessage> messages) {
            this.messages.addAll(messages);
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public MCPPromptResult build() {
            Objects.requireNonNull(promptName, "promptName is required");
            return new MCPPromptResult(
                    promptName, messages, description, metadata, timestamp);
        }
    }

    @Override
    public String toString() {
        return "MCPPromptResult{" +
                "promptName='" + promptName + '\'' +
                ", messageCount=" + messages.size() +
                '}';
    }
}