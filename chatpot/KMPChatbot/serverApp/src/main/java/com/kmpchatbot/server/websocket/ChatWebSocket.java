package com.kmpchatbot.server.websocket;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.kmpchatbot.server.api.dto.ChatResponse;
import com.kmpchatbot.server.service.ChatService;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.websocket.*;
import jakarta.websocket.server.PathParam;
import jakarta.websocket.server.ServerEndpoint;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@ServerEndpoint("/ws/chat/{userId}")
@ApplicationScoped
public class ChatWebSocket {

    private static final Logger LOG = Logger.getLogger(ChatWebSocket.class);
    private static final Map<String, Session> sessions = new ConcurrentHashMap<>();
    
    @Inject
    ChatService chatService;

    private final ObjectMapper objectMapper = new ObjectMapper();

    @OnOpen
    public void onOpen(Session session, @PathParam("userId") String userId) {
        LOG.infof("WebSocket connection opened for user: %s", userId);
        sessions.put(userId, session);
        
        try {
            sendMessage(session, Map.of(
                    "type", "connected",
                    "message", "Connected to chat server",
                    "timestamp", LocalDateTime.now().toString()
            ));
        } catch (IOException e) {
            LOG.error("Error sending connection confirmation", e);
        }
    }

    @OnClose
    public void onClose(Session session, @PathParam("userId") String userId) {
        LOG.infof("WebSocket connection closed for user: %s", userId);
        sessions.remove(userId);
    }

    @OnError
    public void onError(Session session, @PathParam("userId") String userId, Throwable throwable) {
        LOG.errorf(throwable, "WebSocket error for user: %s", userId);
        sessions.remove(userId);
    }

    @OnMessage
    public void onMessage(String message, Session session, @PathParam("userId") String userId) {
        LOG.debugf("Received message from user %s: %s", userId, message);
        
        try {
            // Parse incoming message
            Map<String, Object> messageData = objectMapper.readValue(message, Map.class);
            String content = (String) messageData.get("content");
            Object conversationIdObj = messageData.get("conversationId");
            Long conversationId = conversationIdObj != null ? 
                    Long.valueOf(conversationIdObj.toString()) : null;
            
            // Send typing indicator
            sendMessage(session, Map.of(
                    "type", "typing",
                    "message", "AI is typing...",
                    "timestamp", LocalDateTime.now().toString()
            ));
            
            // Process message through chat service
            Long userIdLong = Long.valueOf(userId);
            ChatResponse response = chatService.processMessage(content, conversationId, userIdLong);
            
            // Send response back
            sendMessage(session, Map.of(
                    "type", "message",
                    "userMessage", response.getUserMessage(),
                    "assistantMessage", response.getAssistantMessage(),
                    "conversationId", response.getConversationId(),
                    "timestamp", LocalDateTime.now().toString()
            ));
            
        } catch (Exception e) {
            LOG.error("Error processing WebSocket message", e);
            try {
                sendMessage(session, Map.of(
                        "type", "error",
                        "message", "Error processing your message: " + e.getMessage(),
                        "timestamp", LocalDateTime.now().toString()
                ));
            } catch (IOException ioException) {
                LOG.error("Error sending error message", ioException);
            }
        }
    }

    private void sendMessage(Session session, Map<String, Object> message) throws IOException {
        if (session.isOpen()) {
            String json = objectMapper.writeValueAsString(message);
            session.getAsyncRemote().sendText(json);
        }
    }

    public static void broadcast(String message) {
        sessions.values().forEach(session -> {
            if (session.isOpen()) {
                session.getAsyncRemote().sendText(message);
            }
        });
    }
}
