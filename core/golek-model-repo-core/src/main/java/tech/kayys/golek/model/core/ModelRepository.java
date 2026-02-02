package tech.kayys.golek.model.core;

import tech.kayys.golek.api.model.ModelManifest;

import java.util.Optional;

public interface ModelRepository {
    Optional<ModelManifest> findById(String modelId, String tenantId);
}