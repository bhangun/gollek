package tech.kayys.golek.engine.model;

import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

import tech.kayys.golek.engine.model.ModelArtifact;
import tech.kayys.golek.engine.model.ModelDescriptor;
import tech.kayys.golek.api.model.ModelFormat;
import tech.kayys.golek.api.model.ModelManifest;
import tech.kayys.golek.engine.model.ModelRef;
import tech.kayys.golek.model.core.Pageable;
import tech.kayys.golek.model.exception.ArtifactDownloadException;
import tech.kayys.wayang.tenant.TenantId;

/**
 * Repository for model artifacts and metadata
 */
public interface ModelRepository {

    /**
     * Load model manifest by ID
     */
    Optional<ModelManifest> findById(String modelId, TenantId tenantId);

    /**
     * List all models for tenant
     */
    List<ModelManifest> findByTenant(TenantId tenantId, Pageable pageable);

    /**
     * Save or update model manifest
     */
    ModelManifest save(ModelManifest manifest);

    /**
     * Download model artifact to local cache
     */
    Path downloadArtifact(
            ModelManifest manifest,
            ModelFormat format) throws ArtifactDownloadException;

    /**
     * Check if artifact is cached locally
     */
    boolean isCached(String modelId, ModelFormat format);

    /**
     * Evict artifact from local cache
     */
    void evictCache(String modelId, ModelFormat format);

    ModelDescriptor resolve(ModelRef ref);

    ModelArtifact fetch(ModelDescriptor descriptor);

    boolean supports(ModelRef ref);
}