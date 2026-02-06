package com.kmpchatbot.server.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ChatResponse {
    
    private Long id;
    
    private String message;
    
    private String role;
    
    @JsonProperty("session_id")
    private String sessionId;
    
    private LocalDateTime timestamp;
}
