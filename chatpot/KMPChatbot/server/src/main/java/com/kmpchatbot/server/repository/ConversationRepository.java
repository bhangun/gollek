package com.kmpchatbot.server.repository;

import com.kmpchatbot.server.domain.Conversation;
import io.quarkus.hibernate.orm.panache.PanacheRepository;
import jakarta.enterprise.context.ApplicationScoped;

import java.util.Optional;

@ApplicationScoped
public class ConversationRepository implements PanacheRepository<Conversation> {
    
    public Optional<Conversation> findBySessionId(String sessionId) {
        return find("sessionId", sessionId).firstResultOptional();
    }
    
    public Conversation findOrCreate(String sessionId) {
        return findBySessionId(sessionId)
                .orElseGet(() -> {
                    Conversation conversation = new Conversation();
                    conversation.setSessionId(sessionId);
                    conversation.setTitle("Chat Session");
                    persist(conversation);
                    return conversation;
                });
    }
}
