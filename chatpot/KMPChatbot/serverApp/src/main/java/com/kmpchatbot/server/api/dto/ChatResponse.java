package com.kmpchatbot.server.api.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ChatResponse {
    private MessageResponse userMessage;
    private MessageResponse assistantMessage;
    private Long conversationId;
}
