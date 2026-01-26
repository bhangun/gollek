package tech.kayys.golek.model;


import com.fasterxml.jackson.annotation.JsonProperty;

public record ChatMessage(
    String role,
    String content,
    @JsonProperty("tool_calls") ToolCall[] toolCalls
) {
    public ChatMessage(String role, String content) {
        this(role, content, null);
    }
    
    public record ToolCall(
        String id,
        String type,
        FunctionCall function
    ) {}
    
    public record FunctionCall(
        String name,
        String arguments
    ) {}
}
