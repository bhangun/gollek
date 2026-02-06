package com.kmpchatbot.server.websocket;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.kmpchatbot.server.dto.ChatRequest;
import com.kmpchatbot.server.dto.ChatResponse;
import com.kmpchatbot.server.service.ChatService;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.websocket.*;
import jakarta.websocket.server.ServerEndpoint;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@ServerEndpoint("/ws/chat")
@ApplicationScoped
public class ChatWebSocket {
    
    private static final Logger LOG = Logger.getLogger(ChatWebSocket.class);
    private static final Map<String, Session> sessions = new ConcurrentHashMap<>();
    
    @Inject
    ChatService chatService;
    
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    @OnOpen
    public void onOpen(Session session) {
        sessions.put(session.getId(), session);
        LOG.infof("WebSocket connection opened: %s", session.getId());
        sendMessage(session, new StatusMessage("connected", "WebSocket connected"));
    }
    
    @OnClose
    public void onClose(Session session) {
        sessions.remove(session.getId());
        LOG.infof("WebSocket connection closed: %s", session.getId());
    }
    
    @OnError
    public void onError(Session session, Throwable throwable) {
        LOG.errorf(throwable, "WebSocket error for session: %s", session.getId());
        sendMessage(session, new StatusMessage("error", throwable.getMessage()));
    }
    
    @OnMessage
    public void onMessage(String message, Session session) {
        LOG.infof("Received WebSocket message from %s: %s", session.getId(), message);
        
        try {
            // Parse incoming message
            ChatRequest request = objectMapper.readValue(message, ChatRequest.class);
            
            // Send typing indicator
            sendMessage(session, new StatusMessage("typing", "AI is thinking..."));
            
            // Process message
            ChatResponse response = chatService.sendMessage(request);
            
            // Send response
            sendMessage(session, response);
            
        } catch (Exception e) {
            LOG.errorf(e, "Error processing WebSocket message");
            sendMessage(session, new StatusMessage("error", "Failed to process message: " + e.getMessage()));
        }
    }
    
    private void sendMessage(Session session, Object message) {
        try {
            String json = objectMapper.writeValueAsString(message);
            session.getAsyncRemote().sendText(json);
        } catch (IOException e) {
            LOG.errorf(e, "Error sending WebSocket message");
        }
    }
    
    public void broadcast(Object message) {
        String json;
        try {
            json = objectMapper.writeValueAsString(message);
        } catch (IOException e) {
            LOG.errorf(e, "Error serializing broadcast message");
            return;
        }
        
        sessions.values().forEach(session -> {
            try {
                session.getAsyncRemote().sendText(json);
            } catch (Exception e) {
                LOG.errorf(e, "Error broadcasting to session: %s", session.getId());
            }
        });
    }
    
    // Helper class for status messages
    public static class StatusMessage {
        public String type;
        public String message;
        
        public StatusMessage(String type, String message) {
            this.type = type;
            this.message = message;
        }
    }
}
