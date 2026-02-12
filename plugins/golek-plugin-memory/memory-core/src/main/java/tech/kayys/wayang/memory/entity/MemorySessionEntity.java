package tech.kayys.wayang.memory.entity;

import io.quarkus.hibernate.reactive.panache.PanacheEntityBase;
import jakarta.persistence.*;
import java.time.Instant;
import java.util.List;
import java.util.Map;

@Entity
@Table(name = "memory_sessions")
public class MemorySessionEntity extends PanacheEntityBase {
    
    @Id
    @Column(name = "session_id")
    public String sessionId;
    
    @Column(name = "user_id", nullable = false)
    public String userId;
    
    @ElementCollection
    @CollectionTable(name = "session_metadata", joinColumns = @JoinColumn(name = "session_id"))
    @MapKeyColumn(name = "key")
    @Column(name = "value")
    public Map<String, String> metadata;
    
    @Column(name = "created_at", nullable = false)
    public Instant createdAt;
    
    @Column(name = "updated_at", nullable = false)
    public Instant updatedAt;
    
    @Column(name = "expires_at")
    public Instant expiresAt;
    
    @OneToMany(mappedBy = "sessionId", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    public List<ConversationMemoryEntity> memories;

    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }
    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public Map<String, String> getMetadata() { return metadata; }
    public void setMetadata(Map<String, String> metadata) { this.metadata = metadata; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(Instant updatedAt) { this.updatedAt = updatedAt; }
    public Instant getExpiresAt() { return expiresAt; }
    public void setExpiresAt(Instant expiresAt) { this.expiresAt = expiresAt; }
    public List<ConversationMemoryEntity> getMemories() { return memories; }
    public void setMemories(List<ConversationMemoryEntity> memories) { this.memories = memories; }
}