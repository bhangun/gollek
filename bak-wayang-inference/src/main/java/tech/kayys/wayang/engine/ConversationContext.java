package tech.kayys.golek.engine;

import java.util.*;

import tech.kayys.golek.model.ChatMessage;

public class ConversationContext {
    private final String id;
    private final List<ChatMessage> messages;
    private final List<Integer> tokens;
    private final long createdAt;
    private long lastAccessedAt;
    
    public ConversationContext(String id) {
        this.id = id;
        this.messages = new ArrayList<>();
        this.tokens = new ArrayList<>();
        this.createdAt = System.currentTimeMillis();
        this.lastAccessedAt = createdAt;
    }
    
    public void addMessage(ChatMessage message, List<Integer> messageTokens) {
        messages.add(message);
        tokens.addAll(messageTokens);
        lastAccessedAt = System.currentTimeMillis();
    }
    
    public void truncate(int maxTokens) {
        if (tokens.size() <= maxTokens) return;
        
        int tokensToRemove = tokens.size() - maxTokens;
        tokens.subList(0, tokensToRemove).clear();
        
        // Try to keep message boundaries
        while (!messages.isEmpty() && tokens.size() < maxTokens / 2) {
            messages.remove(0);
        }
    }
    
    public String getId() { return id; }
    public List<ChatMessage> getMessages() { return Collections.unmodifiableList(messages); }
    public List<Integer> getTokens() { return Collections.unmodifiableList(tokens); }
    public long getCreatedAt() { return createdAt; }
    public long getLastAccessedAt() { return lastAccessedAt; }
}
