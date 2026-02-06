package tech.kayys.golek.spi;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import org.jetbrains.annotations.Nullable;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import tech.kayys.golek.spi.tool.ToolCall;

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

    @NotNull
    private final Role role;

    @NotBlank
    private final String content;

    @Nullable
    private final String name;

    @Nullable
    private final List<ToolCall> toolCalls;

    @Nullable
    private final String toolCallId;

    @JsonCreator
    public Message(
            @JsonProperty("role") Role role,
            @JsonProperty("content") String content,
            @JsonProperty("name") String name,
            @JsonProperty("toolCalls") List<ToolCall> toolCalls,
            @JsonProperty("toolCallId") String toolCallId) {
        this.role = Objects.requireNonNull(role, "role");
        this.content = content; // Content can be null for assistant messages with tool calls
        this.name = name;
        this.toolCalls = toolCalls;
        this.toolCallId = toolCallId;
    }

    public Message(Role role, String content) {
        this(role, content, null, null, null);
    }

    public Role getRole() {
        return role;
    }

    public String getContent() {
        return content;
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

    // Factory methods
    public static Message system(String content) {
        return new Message(Role.SYSTEM, content);
    }

    public static Message user(String content) {
        return new Message(Role.USER, content);
    }

    public static Message assistant(String content) {
        return new Message(Role.ASSISTANT, content);
    }

    public static Message tool(String toolCallId, String content) {
        return new Message(Role.TOOL, content, null, null, toolCallId);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof Message message))
            return false;
        return role == message.role &&
                Objects.equals(content, message.content) &&
                Objects.equals(name, message.name) &&
                Objects.equals(toolCalls, message.toolCalls) &&
                Objects.equals(toolCallId, message.toolCallId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(role, content, name, toolCalls, toolCallId);
    }

    @Override
    public String toString() {
        return "Message{role=" + role + ", content='" +
                (content != null && content.length() > 50 ? content.substring(0, 50) + "..." : content) + "'}";
    }
}