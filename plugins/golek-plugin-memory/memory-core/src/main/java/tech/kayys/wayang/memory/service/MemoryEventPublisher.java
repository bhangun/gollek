package tech.kayys.wayang.memory.service;

import tech.kayys.wayang.memory.model.AgentResponse;
import tech.kayys.wayang.memory.model.MemoryContext;
import io.smallrye.mutiny.Uni;
import io.smallrye.reactive.messaging.MutinyEmitter;
import jakarta.enterprise.context.ApplicationScoped;
import java.util.Map;
import org.eclipse.microprofile.reactive.messaging.Channel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Publishes memory-related events to messaging channels
 */
@ApplicationScoped
public class MemoryEventPublisher {
    
    private static final Logger LOG = LoggerFactory.getLogger(MemoryEventPublisher.class);
    
    @Channel("memory-updated")
    MutinyEmitter<MemoryContext> updatedEmitter;
    
    @Channel("execution-stored")
    MutinyEmitter<AgentResponse> executionEmitter;

    @Channel("backup-scheduled")
    MutinyEmitter<java.util.Map<String, Object>> backupEmitter;

    public Uni<Void> publishMemoryUpdated(MemoryContext context) {
        LOG.debug("Publishing memory updated event for session: {}", context.getSessionId());
        return updatedEmitter.send(context);
    }

    public Uni<Void> publishExecutionStored(String sessionId, AgentResponse response) {
        LOG.debug("Publishing execution stored event for session: {}", sessionId);
        return executionEmitter.send(response);
    }

    public Uni<Void> publishBackupScheduled(String userId, java.time.Duration interval) {
        LOG.debug("Publishing backup scheduled event for user: {} with interval: {}", userId, interval);
        Map<String, Object> event = Map.of(
            "userId", userId,
            "interval", interval.toString(),
            "timestamp", java.time.Instant.now().toString()
        );
        return backupEmitter.send(event);
    }
}
