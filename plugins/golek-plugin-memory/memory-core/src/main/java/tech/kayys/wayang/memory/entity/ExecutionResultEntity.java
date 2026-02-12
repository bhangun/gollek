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

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }
    public String getRequestId() { return requestId; }
    public void setRequestId(String requestId) { this.requestId = requestId; }
    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }
    public ResponseType getType() { return type; }
    public void setType(ResponseType type) { this.type = type; }
    public ResponseStatus getStatus() { return status; }
    public void setStatus(ResponseStatus status) { this.status = status; }
    public Map<String, String> getMetadata() { return metadata; }
    public void setMetadata(Map<String, String> metadata) { this.metadata = metadata; }
    public List<String> getToolCalls() { return toolCalls; }
    public void setToolCalls(List<String> toolCalls) { this.toolCalls = toolCalls; }
    public Instant getTimestamp() { return timestamp; }
    public void setTimestamp(Instant timestamp) { this.timestamp = timestamp; }
}