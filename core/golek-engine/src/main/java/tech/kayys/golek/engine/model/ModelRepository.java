package tech.kayys.golek.engine.model;

import tech.kayys.golek.model.ModelManifest;
import tech.kayys.golek.api.tenant.TenantId;

import java.util.Optional;

public interface ModelRepository {
    Optional<ModelManifest> findById(String modelId, TenantId tenantId);
}