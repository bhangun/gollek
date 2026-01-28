package tech.kayys.golek.provider.openai;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.List;

/**
 * OpenAI message DTO
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class OpenAIMessage {

    private String role;
    private String content; // Can be string or array for multimodal, simplified to string for now
    private String name;
    private List<OpenAIToolCall> toolCalls;
    private String toolCallId;

    public OpenAIMessage() {
    }

    public OpenAIMessage(String role, String content) {
        this.role = role;
        this.content = content;
    }

    public String getRole() {
        return role;
    }

    public void setRole(String role) {
        this.role = role;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public List<OpenAIToolCall> getToolCalls() {
        return toolCalls;
    }

    public void setToolCalls(List<OpenAIToolCall> toolCalls) {
        this.toolCalls = toolCalls;
    }

    public String getToolCallId() {
        return toolCallId;
    }

    public void setToolCallId(String toolCallId) {
        this.toolCallId = toolCallId;
    }
}