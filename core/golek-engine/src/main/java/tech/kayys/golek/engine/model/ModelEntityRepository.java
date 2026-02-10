package tech.kayys.golek.engine.model;

import io.quarkus.hibernate.reactive.panache.PanacheRepositoryBase;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;

import java.util.List;
import java.util.UUID;

@ApplicationScoped
public class ModelEntityRepository implements PanacheRepositoryBase<Model, UUID> {

    public Uni<Model> findByTenantAndModelId(String tenantId, String modelId) {
        return find("tenant.tenantId = ?1 and modelId = ?2", tenantId, modelId).firstResult();
    }

    public Uni<List<Model>> findByTenant(String tenantId) {
        return list("tenant.tenantId", tenantId);
    }

    public Uni<List<Model>> findByStage(Model.ModelStage stage) {
        return list("stage", stage);
    }

    public Uni<Model> findById(UUID id) {
        return find("id", id).firstResult();
    }
}