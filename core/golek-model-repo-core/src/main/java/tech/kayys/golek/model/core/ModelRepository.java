package tech.kayys.golek.model.core;

import tech.kayys.golek.spi.model.ModelManifest;

import io.smallrye.mutiny.Uni;

import java.util.List;

public interface ModelRepository {
    Uni<ModelManifest> findById(String modelId, String tenantId);

    Uni<List<ModelManifest>> list(String tenantId, Pageable pageable);

    Uni<ModelManifest> save(ModelManifest manifest);

    Uni<Void> delete(String modelId, String tenantId);
}