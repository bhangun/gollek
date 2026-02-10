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
}