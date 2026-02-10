package tech.kayys.wayang.memory.messaging;

import tech.kayys.wayang.memory.model.AgentResponse;
import tech.kayys.wayang.memory.service.MemoryService;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.reactive.messaging.Incoming;
import org.eclipse.microprofile.reactive.messaging.Message;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@ApplicationScoped
public class MemoryMessageConsumer {
    
    private static final Logger LOG = LoggerFactory.getLogger(MemoryMessageConsumer.class);
    
    @Inject
    MemoryService memoryService;

    @Incoming("execution-results")
    public Uni<Void> processExecutionResult(Message<AgentResponse> message) {
        AgentResponse result = message.getPayload();
        LOG.info("Received execution result: {} for session: {}", result.getId(), result.getSessionId());
        
        return memoryService.storeExecutionResult(result.getSessionId(), result)
            .onItem().invoke(() -> LOG.info("Stored execution result: {}", result.getId()))
            .onFailure().invoke(throwable -> 
                LOG.error("Failed to store execution result: {}", result.getId(), throwable))
            .onItemOrFailure().transformToUni((unused, throwable) -> {
                if (throwable != null) {
                    return message.nack(throwable);
                } else {
                    return message.ack();
                }
            });
    }

    @Incoming("planning-requests")
    public Uni<Void> processPlanningRequest(Message<PlanningRequest> message) {
        PlanningRequest request = message.getPayload();
        LOG.info("Received planning request for session: {}", request.sessionId);
        
        return memoryService.getContext(request.sessionId, request.userId)
            .onItem().transformToUni(context -> 
                publishMemoryContext(request.requestId, context)
            )
            .onItemOrFailure().transformToUni((unused, throwable) -> {
                if (throwable != null) {
                    LOG.error("Failed to process planning request: {}", request.requestId, throwable);
                    return message.nack(throwable);
                } else {
                    return message.ack();
                }
            });
    }

    private Uni<Void> publishMemoryContext(String requestId, tech.kayys.wayang.memory.model.MemoryContext context) {
        // This would publish the memory context back to the requesting service
        LOG.info("Publishing memory context for request: {}", requestId);
        return Uni.createFrom().voidItem();
    }

    public static class PlanningRequest {
        public String requestId;
        public String sessionId;
        public String userId;
        public String task;
    }
}