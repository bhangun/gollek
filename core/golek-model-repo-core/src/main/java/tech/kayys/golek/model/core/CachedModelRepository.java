package tech.kayys.golek.model.core;

import io.quarkus.cache.CacheResult;

import java.util.Optional;

import io.quarkus.cache.CacheKey;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.golek.api.model.ModelFormat;
import tech.kayys.golek.api.tenant.TenantId;
import tech.kayys.golek.model.repository.ModelRepository;

import java.nio.file.Path;
import java.util.List;

/**
 * Model manifest caching for faster lookups.
 */
@ApplicationScoped
public class CachedModelRepository implements ModelRepository {

    @Inject
    ModelRepository delegate; // Use interface instead of concrete Postgres implementation

    @Override
    @CacheResult(cacheName = "model-manifests")
    public Optional<ModelManifest> findById(
            @CacheKey String modelId,
            @CacheKey TenantId tenantId) {
        return delegate.findById(modelId, tenantId);
    }

    @Override
    public List<ModelManifest> findByTenant(TenantId tenantId, Pageable pageable) {
        return delegate.findByTenant(tenantId, pageable);
    }

    @Override
    public ModelManifest save(ModelManifest manifest) {
        return delegate.save(manifest);
    }

    @Override
    public Path downloadArtifact(ModelManifest manifest, ModelFormat format)
            throws tech.kayys.golek.model.exception.ArtifactDownloadException {
        return delegate.downloadArtifact(manifest, format);
    }

    @Override
    public boolean isCached(String modelId, ModelFormat format) {
        return delegate.isCached(modelId, format);
    }

    @Override
    public void evictCache(String modelId, ModelFormat format) {
        delegate.evictCache(modelId, format);
    }

    @Override
    public ModelDescriptor resolve(ModelRef ref) {
        return delegate.resolve(ref);
    }

    @Override
    public ModelArtifact fetch(ModelDescriptor descriptor) {
        return delegate.fetch(descriptor);
    }

    @Override
    public boolean supports(ModelRef ref) {
        return delegate.supports(ref);
    }
}