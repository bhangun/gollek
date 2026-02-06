package com.kmpchatbot.server.api.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;

// User DTOs
@Data
@NoArgsConstructor
@AllArgsConstructor
public class UserRequest {
    @NotBlank(message = "Username is required")
    private String username;

    @NotBlank(message = "Password is required")
    private String password;

    @NotBlank(message = "Email is required")
    private String email;

    private String fullName;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class UserResponse {
    private Long id;
    private String username;
    private String email;
    private String fullName;
    private LocalDateTime createdAt;
    private boolean active;
}

// Authentication DTOs
@Data
@NoArgsConstructor
@AllArgsConstructor
class LoginRequest {
    @NotBlank(message = "Username is required")
    private String username;

    @NotBlank(message = "Password is required")
    private String password;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class LoginResponse {
    private String token;
    private UserResponse user;
}

// Conversation DTOs
@Data
@NoArgsConstructor
@AllArgsConstructor
class ConversationRequest {
    private String title;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class ConversationResponse {
    private Long id;
    private String title;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private int messageCount;
    private boolean archived;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class ConversationDetailResponse {
    private Long id;
    private String title;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private List<MessageResponse> messages;
    private boolean archived;
}

// Message DTOs
@Data
@NoArgsConstructor
@AllArgsConstructor
class MessageRequest {
    @NotBlank(message = "Content is required")
    private String content;

    @NotNull(message = "Conversation ID is required")
    private Long conversationId;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class MessageResponse {
    private Long id;
    private String content;
    private String role;
    private LocalDateTime timestamp;
    private String modelUsed;
}

// Chat Request/Response (for AI interaction)
@Data
@NoArgsConstructor
@AllArgsConstructor
class ChatRequest {
    @NotBlank(message = "Message is required")
    private String message;

    private Long conversationId;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class ChatResponse {
    private MessageResponse userMessage;
    private MessageResponse assistantMessage;
    private Long conversationId;
}

// WebSocket DTOs
@Data
@NoArgsConstructor
@AllArgsConstructor
class WebSocketMessage {
    private String type; // "message", "typing", "connected", "error"
    private String content;
    private Long conversationId;
    private String role;
    private LocalDateTime timestamp;
}

// Error Response
@Data
@NoArgsConstructor
@AllArgsConstructor
class ErrorResponse {
    private String message;
    private String error;
    private int status;
    private LocalDateTime timestamp;

    public ErrorResponse(String message, String error, int status) {
        this.message = message;
        this.error = error;
        this.status = status;
        this.timestamp = LocalDateTime.now();
    }
}

// API Response wrapper
@Data
@NoArgsConstructor
@AllArgsConstructor
class ApiResponse<T> {
    private boolean success;
    private String message;
    private T data;

    public static <T> ApiResponse<T> success(T data) {
        return new ApiResponse<>(true, "Success", data);
    }

    public static <T> ApiResponse<T> success(String message, T data) {
        return new ApiResponse<>(true, message, data);
    }

    public static <T> ApiResponse<T> error(String message) {
        return new ApiResponse<>(false, message, null);
    }
}

// Make classes public
public class ChatbotDtos {
    // Container class for all DTOs
}
