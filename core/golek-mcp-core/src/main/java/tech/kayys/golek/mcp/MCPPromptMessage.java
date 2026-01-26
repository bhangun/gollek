package tech.kayys.golek.provider.core.mcp;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.*;

/**
 * Message content from a prompt execution.
 */
public final class MCPPromptMessage {

    public enum Role {
        USER, ASSISTANT, SYSTEM
    }

    private final Role role;
    private final List<Content> content;

    @JsonCreator
    public MCPPromptMessage(
            @JsonProperty("role") Role role,
            @JsonProperty("content") List<Content> content) {
        this.role = Objects.requireNonNull(role, "role");
        this.content = content != null
                ? Collections.unmodifiableList(new ArrayList<>(content))
                : Collections.emptyList();
    }

    // Content type for prompt messages
    public record Content(
            @JsonProperty("type") String type, // text|image|resource
            @JsonProperty("text") String text,
            @JsonProperty("data") String data,
            @JsonProperty("mimeType") String mimeType) {
        public Content {
            Objects.requireNonNull(type, "type");
        }

        public static Content text(String text) {
            return new Content("text", text, null, null);
        }

        public static Content image(String data, String mimeType) {
            return new Content("image", null, data, mimeType);
        }
    }

    // Getters
    public Role getRole() {
        return role;
    }

    public List<Content> getContent() {
        return content;
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
     * Get all text concatenated
     */
    public String getAllText() {
        return content.stream()
                .filter(c -> "text".equals(c.type()))
                .map(Content::text)
                .filter(Objects::nonNull)
                .reduce("", (a, b) -> a + "\n" + b)
                .trim();
    }

    @Override
    public String toString() {
        return "MCPPromptMessage{role=" + role + ", contentCount=" + content.size() + '}';
    }
}