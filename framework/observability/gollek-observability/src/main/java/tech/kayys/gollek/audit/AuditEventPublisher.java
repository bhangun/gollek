package tech.kayys.gollek.audit;

import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.observability.AuditPayload;

/**
 * Interface for publishing audit events to external systems (e.g., Kafka,
 * EventBridge).
 */
public interface AuditEventPublisher {

    /**
     * Publishes an audit payload.
     * 
     * @param payload the audit payload to publish
     * @return a Uni representing the asynchronous completion
     */
    Uni<Void> publish(AuditPayload payload);
}
