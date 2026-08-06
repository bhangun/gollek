package tech.kayys.gollek.spi;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

import org.jetbrains.annotations.Nullable;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import tech.kayys.gollek.spi.tool.ToolCall;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Immutable message in a conversation.
 */
public final class Message {

    public enum Role {
        SYSTEM,
        USER,
        ASSISTANT,
        FUNCTION,
        TOOL
    }

    /**
     * Multi-modal content part type discriminator.
     */
    public enum ContentType {
        TEXT, IMAGE_URL, IMAGE_BASE64, AUDIO_URL, AUDIO_BASE64, DOCUMENT
    }

    /**
     * A single typed content part inside a multi-modal message.
     *
     * <p>For text-only messages use {@link #content} directly.
     * For vision/audio messages populate {@link #contentParts} instead.
     *
     * <h3>Example (vision request)</h3>
     * <pre>
     * Message.multiModal(Role.USER, List.of(
     *   ContentPart.text("What is in this image?"),
     *   ContentPart.imageUrl("https://example.com/photo.jpg")
     * ));
     * </pre>
     */
    public record ContentPart(
            @JsonProperty("type")     ContentType type,
            @JsonProperty("text")     @Nullable String text,
            @JsonProperty("url")      @Nullable String url,
            @JsonProperty("data")     @Nullable String data,    // base64
            @JsonProperty("mimeType") @Nullable String mimeType
    ) {
        public static ContentPart text(String text) {
            return new ContentPart(ContentType.TEXT, text, null, null, null);
        }
        public static ContentPart imageUrl(String url) {
            return new ContentPart(ContentType.IMAGE_URL, null, url, null, "image/*");
        }
        public static ContentPart imageBase64(String data, String mimeType) {
            return new ContentPart(ContentType.IMAGE_BASE64, null, null, data, mimeType);
        }
        public static ContentPart audioUrl(String url, String mimeType) {
            return new ContentPart(ContentType.AUDIO_URL, null, url, null, mimeType);
        }
        public static ContentPart audioBase64(String data, String mimeType) {
            return new ContentPart(ContentType.AUDIO_BASE64, null, null, data, mimeType);
        }
    }

    @NotNull
    private final Role role;

    /**
     * Plain text content (backward-compatible, used for text-only messages).
     * Null when {@link #contentParts} is set.
     */
    @Nullable
    private final String content;

    /**
     * Structured multi-modal content parts.
     * Takes precedence over {@link #content} when non-empty.
     */
    @Nullable
    private final List<ContentPart> contentParts;

    @Nullable
    private final String name;

    @Nullable
    private final List<ToolCall> toolCalls;

    @Nullable
    private final String toolCallId;

    @JsonCreator
    public Message(
            @JsonProperty("role")         Role role,
            @JsonProperty("content")      String content,
            @JsonProperty("contentParts") List<ContentPart> contentParts,
            @JsonProperty("name")         String name,
            @JsonProperty("toolCalls")    List<ToolCall> toolCalls,
            @JsonProperty("toolCallId")   String toolCallId) {
        this.role         = Objects.requireNonNull(role, "role");
        this.content      = content;
        this.contentParts = (contentParts != null && !contentParts.isEmpty())
                            ? Collections.unmodifiableList(contentParts)
                            : null;
        this.name         = name;
        this.toolCalls    = toolCalls;
        this.toolCallId   = toolCallId;
    }

    public Message(Role role, String content) {
        this(role, content, null, null, null, null);
    }

    public Role getRole() {
        return role;
    }

    public String getContent() {
        return content;
    }

    /**
     * Returns structured content parts for multi-modal messages.
     * Returns null for plain-text messages.
     */
    @Nullable
    public List<ContentPart> getContentParts() {
        return contentParts;
    }

    /**
     * Returns true when this message carries structured multi-modal content
     * (images, audio, etc.) rather than plain text.
     */
    public boolean isMultiModal() {
        return contentParts != null && !contentParts.isEmpty();
    }

    public String getName() {
        return name;
    }

    public List<ToolCall> getToolCalls() {
        return toolCalls;
    }

    public String getToolCallId() {
        return toolCallId;
    }

    // Factory methods for common roles
    public static Message system(String content) {
        return new Message(Message.Role.SYSTEM, content);
    }

    public static Message user(String content) {
        return new Message(Message.Role.USER, content);
    }

    public static Message assistant(String content) {
        return new Message(Message.Role.ASSISTANT, content);
    }

    public static Message assistantWithToolCalls(String content, List<ToolCall> toolCalls) {
        return new Message(Message.Role.ASSISTANT, content, null, null, toolCalls, null);
    }

    public static Message tool(String toolCallId, String content) {
        return new Message(Message.Role.TOOL, content, null, null, null, toolCallId);
    }

    /**
     * Creates a multi-modal user message with structured content parts.
     *
     * @param role         message role
     * @param contentParts ordered list of content parts (text, image, audio)
     */
    public static Message multiModal(Role role, List<ContentPart> contentParts) {
        return new Message(role, null, contentParts, null, null, null);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Message message)) return false;
        return role == message.role &&
                Objects.equals(content, message.content) &&
                Objects.equals(contentParts, message.contentParts) &&
                Objects.equals(name, message.name) &&
                Objects.equals(toolCalls, message.toolCalls) &&
                Objects.equals(toolCallId, message.toolCallId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(role, content, contentParts, name, toolCalls, toolCallId);
    }

    @Override
    public String toString() {
        return "Message{role=" + role + ", content='" +
                (content != null && content.length() > 50 ? content.substring(0, 50) + "..." : content) + "'}";
    }
}