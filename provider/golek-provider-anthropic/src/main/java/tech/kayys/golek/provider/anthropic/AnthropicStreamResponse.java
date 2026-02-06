package tech.kayys.golek.provider.anthropic;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

/**
 * Represents a response chunk from Anthropic's streaming API.
 * Anthropic uses Server-Sent Events (SSE) for streaming responses.
 */
public class AnthropicStreamResponse {

    private String type;  // e.g., "content_block_start", "content_block_delta", "content_block_stop", "message_start", etc.
    private Integer index; // Index of the content block
    private ContentBlock contentBlock; // Content block for start events
    private Delta delta; // Delta for delta events
    private Message message; // Message for start events
    private Usage usage; // Usage information
    
    // Constructors
    public AnthropicStreamResponse() {}
    
    // Getters and setters
    public String getType() {
        return type;
    }
    
    public void setType(String type) {
        this.type = type;
    }
    
    public Integer getIndex() {
        return index;
    }
    
    public void setIndex(Integer index) {
        this.index = index;
    }
    
    public ContentBlock getContentBlock() {
        return contentBlock;
    }
    
    public void setContentBlock(ContentBlock contentBlock) {
        this.contentBlock = contentBlock;
    }
    
    public Delta getDelta() {
        return delta;
    }
    
    public void setDelta(Delta delta) {
        this.delta = delta;
    }
    
    public Message getMessage() {
        return message;
    }
    
    public void setMessage(Message message) {
        this.message = message;
    }
    
    public Usage getUsage() {
        return usage;
    }
    
    public void setUsage(Usage usage) {
        this.usage = usage;
    }
    
    // Inner classes for nested structures
    public static class ContentBlock {
        private String type; // Usually "text"
        private String text;
        
        public ContentBlock() {}
        
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        
        public String getText() { return text; }
        public void setText(String text) { this.text = text; }
    }
    
    public static class Delta {
        private String type; // Usually "text_delta"
        private String text; // The incremental text
        
        public Delta() {}
        
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        
        public String getText() { return text; }
        public void setText(String text) { this.text = text; }
    }
    
    public static class Message {
        private String id;
        private String type; // Usually "message"
        private String role; // Usually "assistant"
        private List<ContentBlock> content;
        private String model;
        private String stopReason;
        private String stopSequence;
        private Usage usage;
        
        public Message() {}
        
        // Getters and setters
        public String getId() { return id; }
        public void setId(String id) { this.id = id; }
        
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        
        public String getRole() { return role; }
        public void setRole(String role) { this.role = role; }
        
        public List<ContentBlock> getContent() { return content; }
        public void setContent(List<ContentBlock> content) { this.content = content; }
        
        public String getModel() { return model; }
        public void setModel(String model) { this.model = model; }
        
        public String getStopReason() { return stopReason; }
        public void setStopReason(String stopReason) { this.stopReason = stopReason; }
        
        public String getStopSequence() { return stopSequence; }
        public void setStopSequence(String stopSequence) { this.stopSequence = stopSequence; }
        
        public Usage getUsage() { return usage; }
        public void setUsage(Usage usage) { this.usage = usage; }
    }
    
    public static class Usage {
        @JsonProperty("input_tokens")
        private Integer inputTokens;
        
        @JsonProperty("output_tokens")
        private Integer outputTokens;
        
        public Usage() {}
        
        public Integer getInputTokens() { return inputTokens; }
        public void setInputTokens(Integer inputTokens) { this.inputTokens = inputTokens; }
        
        public Integer getOutputTokens() { return outputTokens; }
        public void setOutputTokens(Integer outputTokens) { this.outputTokens = outputTokens; }
    }
}