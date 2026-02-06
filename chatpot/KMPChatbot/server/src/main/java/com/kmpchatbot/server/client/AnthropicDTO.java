package com.kmpchatbot.server.client;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

public class AnthropicDTO {
    
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class AnthropicRequest {
        private String model;
        
        @JsonProperty("max_tokens")
        private Integer maxTokens;
        
        private List<AnthropicMessage> messages;
    }
    
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class AnthropicMessage {
        private String role;
        private String content;
    }
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class AnthropicResponse {
        private String id;
        private String type;
        private String role;
        private List<ContentBlock> content;
        private String model;
        
        @JsonProperty("stop_reason")
        private String stopReason;
    }
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ContentBlock {
        private String type;
        private String text;
    }
}
