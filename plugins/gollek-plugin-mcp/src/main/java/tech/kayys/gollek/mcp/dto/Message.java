package tech.kayys.gollek.mcp.dto;

import java.util.Map;

/**
 * Represents a message in an MCP conversation.
 */
public class Message {

    private final String role;
    private final String content;
    private final Map<String, Object> additionalProperties;

    public Message(String role, String content, Map<String, Object> additionalProperties) {
        this.role = role;
        this.content = content;
        this.additionalProperties = additionalProperties != null ? additionalProperties : Map.of();
    }

    public String getRole() {
        return role;
    }

    public String getContent() {
        return content;
    }

    public Map<String, Object> getAdditionalProperties() {
        return additionalProperties;
    }

    public static Message of(String role, String content) {
        return new Message(role, content, null);
    }
}