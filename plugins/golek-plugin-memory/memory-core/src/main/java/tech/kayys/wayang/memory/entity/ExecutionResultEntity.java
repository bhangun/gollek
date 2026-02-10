package tech.kayys.wayang.memory.entity;

import tech.kayys.wayang.memory.model.ResponseType;
import tech.kayys.wayang.memory.model.ResponseStatus;
import io.quarkus.hibernate.reactive.panache.PanacheEntityBase;
import jakarta.persistence.*;
import java.time.Instant;
import java.util.List;
import java.util.Map;

@Entity
@Table(name = "execution_results")
public class ExecutionResultEntity extends PanacheEntityBase {
    
    @Id
    public String id;
    
    @Column(name = "session_id", nullable = false)
    public String sessionId;
    
    @Column(name = "request_id")
    public String requestId;
    
    @Column(name = "content", columnDefinition = "TEXT")
    public String content;
    
    @Enumerated(EnumType.STRING)
    @Column(name = "type")
    public ResponseType type;
    
    @Enumerated(EnumType.STRING)
    @Column(name = "status")
    public ResponseStatus status;
    
    @ElementCollection
    @CollectionTable(name = "execution_metadata", joinColumns = @JoinColumn(name = "execution_id"))
    @MapKeyColumn(name = "key")
    @Column(name = "value")
    public Map<String, String> metadata;
    
    @ElementCollection
    @CollectionTable(name = "execution_tool_calls", joinColumns = @JoinColumn(name = "execution_id"))
    @Column(name = "tool_call")
    public List<String> toolCalls;
    
    @Column(name = "timestamp", nullable = false)
    public Instant timestamp;
}