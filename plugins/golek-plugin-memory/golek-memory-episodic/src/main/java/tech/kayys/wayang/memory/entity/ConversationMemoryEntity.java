package tech.kayys.wayang.memory.entity;

import io.quarkus.hibernate.reactive.panache.PanacheEntityBase;
import jakarta.persistence.*;
import java.time.Instant;
import java.util.List;
import java.util.Map;

@Entity
@Table(name = "conversation_memories")
public class ConversationMemoryEntity extends PanacheEntityBase {
    
    @Id
    public String id;
    
    @Column(name = "session_id", nullable = false)
    public String sessionId;
    
    @Column(name = "role", nullable = false)
    public String role;
    
    @Column(name = "content", columnDefinition = "TEXT")
    public String content;
    
    @ElementCollection
    @CollectionTable(name = "memory_metadata", joinColumns = @JoinColumn(name = "memory_id"))
    @MapKeyColumn(name = "key")
    @Column(name = "value")
    public Map<String, String> metadata;
    
    @ElementCollection
    @CollectionTable(name = "memory_embeddings", joinColumns = @JoinColumn(name = "memory_id"))
    @Column(name = "embedding_value")
    public List<Double> embedding;
    
    @Column(name = "timestamp", nullable = false)
    public Instant timestamp;
    
    @Column(name = "relevance_score")
    public Double relevanceScore;
    
    @Column(name = "is_summary")
    public Boolean isSummary = false;
    
    @Column(name = "summary_of_session")
    public String summaryOfSession;
}
