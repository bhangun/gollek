package tech.kayys.golek.provider.core.mcp;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;
import java.util.*;

/**
 * Result of an MCP tool execution.
 * Supports both success and error states.
 */
public final class MCPToolResult {

    private final String toolName;
    private final boolean success;
    private final List<Content> content;
    private final String errorMessage;
    private final Map<String, Object> metadata;
    private final Instant timestamp;

    @JsonCreator
    public MCPToolResult(
            @JsonProperty("toolName") String toolName,
            @JsonProperty("success") boolean success,
            @JsonProperty("content") List<Content> content,
            @JsonProperty("errorMessage") String errorMessage,
            @JsonProperty("metadata") Map<String, Object> metadata,
            @JsonProperty("timestamp") Instant timestamp) {
        this.toolName = Objects.requireNonNull(toolName, "toolName");
        this.success = success;
        this.content = content != null
                ? Collections.unmodifiableList(new ArrayList<>(content))
                : Collections.emptyList();
        this.errorMessage = errorMessage;
        this.metadata = metadata != null
                ? Collections.unmodifiableMap(new HashMap<>(metadata))
                : Collections.emptyMap();
        this.timestamp = timestamp != null ? timestamp : Instant.now();
    }

    // Content type for tool results
    public record Content(
            @JsonProperty("type") String type, // text|image|resource
            @JsonProperty("text") String text,
            @JsonProperty("data") String data,
            @JsonProperty("mimeType") String mimeType,
            @JsonProperty("uri") String uri) {
        public Content {
            Objects.requireNonNull(type, "type");
        }

        public static Content text(String text) {
            return new Content("text", text, null, null, null);
        }

        public static Content image(String data, String mimeType) {
            return new Content("image", null, data, mimeType, null);
        }

        public static Content resource(String uri) {
            return new Content("resource", null, null, null, uri);
        }
    }

    // Getters
    public String getToolName() {
        return toolName;
    }

    public boolean isSuccess() {
        return success;
    }

    public List<Content> getContent() {
        return content;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public Instant getTimestamp() {
        return timestamp;
    }

    /**
     * Get first text content
     */
    public Optional<String> getTextContent() {
        return content.stream()
                .filter(c -> "text".equals(c.type()))
                .map(Content::text)
                .findFirst();
    }

    /**
     * Get all text content concatenated
     */
    public String getAllText() {
        return content.stream()
                .filter(c -> "text".equals(c.type()))
                .map(Content::text)
                .filter(Objects::nonNull)
                .reduce("", (a, b) -> a + "\n" + b)
                .trim();
    }

    // Factory methods
    public static Builder builder() {
        return new Builder();
    }

    public static MCPToolResult success(String toolName, List<Content> content) {
        return builder()
                .toolName(toolName)
                .success(true)
                .content(content)
                .build();
    }

    public static MCPToolResult success(String toolName, String textContent) {
        return success(toolName, List.of(Content.text(textContent)));
    }

    public static MCPToolResult error(String toolName, String errorMessage) {
        return builder()
                .toolName(toolName)
                .success(false)
                .errorMessage(errorMessage)
                .build();
    }

    public static class Builder {
        private String toolName;
        private boolean success = true;
        private final List<Content> content = new ArrayList<>();
        private String errorMessage;
        private final Map<String, Object> metadata = new HashMap<>();
        private Instant timestamp = Instant.now();

        public Builder toolName(String toolName) {
            this.toolName = toolName;
            return this;
        }

        public Builder success(boolean success) {
            this.success = success;
            return this;
        }

        public Builder content(List<Content> content) {
            this.content.addAll(content);
            return this;
        }

        public Builder addContent(Content content) {
            this.content.add(content);
            return this;
        }

        public Builder textContent(String text) {
            this.content.add(Content.text(text));
            return this;
        }

        public Builder errorMessage(String errorMessage) {
            this.errorMessage = errorMessage;
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

        public MCPToolResult build() {
            return new MCPToolResult(
                    toolName, success, content, errorMessage, metadata, timestamp);
        }
    }

    @Override
    public String toString() {
        return "MCPToolResult{" +
                "toolName='" + toolName + '\'' +
                ", success=" + success +
                ", contentCount=" + content.size() +
                '}';
    }
}