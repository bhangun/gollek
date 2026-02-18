package tech.kayys.gollek.observability;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.AuditPayload;
import io.smallrye.mutiny.Uni;
import io.quarkus.arc.DefaultBean;

@ApplicationScoped
@DefaultBean
public class DefaultAuditEventPublisher implements AuditEventPublisher {

    @Override
    public Uni<Void> publish(AuditPayload payload) {
        // Default no-op implementation
        return Uni.createFrom().voidItem();
    }
}
