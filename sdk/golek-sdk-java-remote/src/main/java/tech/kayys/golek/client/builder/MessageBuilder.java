package tech.kayys.golek.client.builder;

import tech.kayys.golek.api.Message;
import tech.kayys.golek.api.tool.ToolCall;

import java.util.List;

/**
 * Utility class for building Message objects.
 */
public class MessageBuilder {
    
    private Message.Role role;
    private String content;
    private String name;
    private List<ToolCall> toolCalls;
    private String toolCallId;
    
    public static MessageBuilder builder() {
        return new MessageBuilder();
    }
    
    public MessageBuilder role(Message.Role role) {
        this.role = role;
        return this;
    }
    
    public MessageBuilder content(String content) {
        this.content = content;
        return this;
    }
    
    public MessageBuilder name(String name) {
        this.name = name;
        return this;
    }
    
    public MessageBuilder toolCalls(List<ToolCall> toolCalls) {
        this.toolCalls = toolCalls;
        return this;
    }
    
    public MessageBuilder toolCallId(String toolCallId) {
        this.toolCallId = toolCallId;
        return this;
    }
    
    public Message build() {
        return new Message(role, content, name, toolCalls, toolCallId);
    }
    
    // Convenience methods for common roles
    public static Message system(String content) {
        return new Message(Message.Role.SYSTEM, content);
    }
    
    public static Message user(String content) {
        return new Message(Message.Role.USER, content);
    }
    
    public static Message assistant(String content) {
        return new Message(Message.Role.ASSISTANT, content);
    }
    
    public static Message tool(String toolCallId, String content) {
        return new Message(Message.Role.TOOL, content, null, null, toolCallId);
    }
}