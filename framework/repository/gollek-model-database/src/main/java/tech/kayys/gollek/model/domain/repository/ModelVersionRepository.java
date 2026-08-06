package tech.kayys.gollek.registry.repository;

import io.quarkus.hibernate.reactive.panache.PanacheRepository;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.registry.model.ModelVersion;

import java.util.List;
import java.util.UUID;

@ApplicationScoped
public class ModelVersionRepository implements PanacheRepository<ModelVersion> {

    public Uni<ModelVersion> findByModelAndVersion(UUID modelId, String version) {
        return find("model.id = ?1 and version = ?2", modelId, version).firstResult();
    }

    public Uni<List<ModelVersion>> findActiveVersions(UUID modelId) {
        return list("model.id = ?1 and status = ?2 order by createdAt desc",
                modelId, ModelVersion.VersionStatus.ACTIVE);
    }
}
