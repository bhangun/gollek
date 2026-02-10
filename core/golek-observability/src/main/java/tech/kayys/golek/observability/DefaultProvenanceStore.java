package tech.kayys.golek.observability;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.golek.spi.AuditPayload;
import io.smallrye.mutiny.Uni;
import io.quarkus.arc.DefaultBean;

@ApplicationScoped
@DefaultBean
public class DefaultProvenanceStore implements ProvenanceStore {

    @Override
    public Uni<Void> store(AuditPayload payload) {
        // Default no-op implementation
        return Uni.createFrom().voidItem();
    }
}
