package com.kmpchatbot.server.repository;

import com.kmpchatbot.server.domain.Message;
import io.quarkus.hibernate.orm.panache.PanacheRepository;
import jakarta.enterprise.context.ApplicationScoped;

import java.util.List;

@ApplicationScoped
public class MessageRepository implements PanacheRepository<Message> {
    
    public List<Message> findBySessionId(String sessionId) {
        return list("sessionId = ?1 order by timestamp asc", sessionId);
    }
    
    public List<Message> findByConversationId(Long conversationId) {
        return list("conversation.id = ?1 order by timestamp asc", conversationId);
    }
    
    public void deleteBySessionId(String sessionId) {
        delete("sessionId", sessionId);
    }
}
