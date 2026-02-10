package tech.kayys.golek.engine.model;

import io.quarkus.cache.CacheResult;
import io.quarkus.cache.CacheKey;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.golek.spi.model.ModelFormat;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.spi.model.Pageable;
import tech.kayys.golek.spi.model.ModelArtifact;
import tech.kayys.golek.spi.model.ModelDescriptor;
import tech.kayys.golek.spi.model.ModelRef;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;

@ApplicationScoped
public class CachedModelRepository {

    @jakarta.inject.Inject
    jakarta.enterprise.inject.Instance<ModelEntityRepository> dbRepository;

    @jakarta.inject.Inject
    jakarta.enterprise.inject.Instance<tech.kayys.golek.model.core.ModelRepository> repositories;

    @Inject
    ModelRepositoryRegistry registry;

    @CacheResult(cacheName = "model-manifests")
    public Uni<ModelManifest> findById(
            @CacheKey String modelId,
            @CacheKey String tenantId) {

        // 1. Try all registered repositories first
        List<Uni<ModelManifest>> repositoryLookups = new java.util.ArrayList<>();
        for (var repo : repositories) {
            repositoryLookups.add(repo.findById(modelId, tenantId)
                    .onFailure().recoverWithNull());
        }

        return Uni.combine().all().unis(repositoryLookups).with(results -> {
            for (Object result : results) {
                if (result != null)
                    return (ModelManifest) result;
            }
            return null;
        }).onItem().ifNull().switchTo(() -> {
            // 2. Fallback to database if available
            if (dbRepository.isResolvable()) {
                try {
                    return dbRepository.get().findByTenantAndModelId(tenantId, modelId)
                            .map(m -> m != null ? m.toManifest() : null)
                            .onFailure().recoverWithNull();
                } catch (Exception e) {
                    return Uni.createFrom().nullItem();
                }
            }
            return Uni.createFrom().nullItem();
        });
    }

    public Uni<List<ModelManifest>> list(String tenantId, Pageable pageable) {
        // Collect from all repositories
        List<Uni<List<ModelManifest>>> repositoryLists = new java.util.ArrayList<>();
        for (var repo : repositories) {
            repositoryLists.add(repo.list(tenantId, pageable)
                    .onFailure().recoverWithItem(List.of()));
        }

        if (dbRepository.isResolvable()) {
            repositoryLists.add(dbRepository.get().findByTenant(tenantId)
                    .map(list -> list.stream().map(Model::toManifest).collect(java.util.stream.Collectors.toList()))
                    .onFailure().recoverWithItem(List.of()));
        }

        return Uni.combine().all().unis(repositoryLists).with(results -> {
            java.util.Set<String> seenIds = new java.util.HashSet<>();
            List<ModelManifest> all = new java.util.ArrayList<>();
            for (Object result : results) {
                @SuppressWarnings("unchecked")
                List<ModelManifest> list = (List<ModelManifest>) result;
                for (ModelManifest m : list) {
                    if (seenIds.add(m.modelId())) {
                        all.add(m);
                    }
                }
            }
            return all;
        });
    }

    public Uni<ModelManifest> save(ModelManifest manifest) {
        return Uni.createFrom()
                .failure(new UnsupportedOperationException("Saving via CachedModelRepository not implemented yet"));
    }

    public Uni<Void> delete(String modelId, String tenantId) {
        List<Uni<Void>> deletions = new java.util.ArrayList<>();
        for (var repo : repositories) {
            deletions.add(repo.delete(modelId, tenantId).onFailure().recoverWithNull());
        }

        if (dbRepository.isResolvable()) {
            try {
                // Try to parse as UUID, if fails it's probably a modelId string
                try {
                    UUID uuid = UUID.fromString(modelId);
                    deletions.add(dbRepository.get().deleteById(uuid).replaceWithVoid().onFailure().recoverWithNull());
                } catch (IllegalArgumentException e) {
                    // If not a UUID, we might need a find-then-delete, but let's keep it simple for
                    // now
                }
            } catch (Exception e) {
                // Ignore DB errors
            }
        }

        return Uni.join().all(deletions).andCollectFailures().replaceWithVoid();
    }

    public Path downloadArtifact(ModelManifest manifest, ModelFormat format)
            throws tech.kayys.golek.model.exception.ArtifactDownloadException {
        for (var repo : repositories) {
            try {
                if (repo.isCached(manifest.modelId(), format)) {
                    return repo.downloadArtifact(manifest, format);
                }
            } catch (Exception e) {
                // Ignore and try next
            }
        }
        return null;
    }

    public boolean isCached(String modelId, ModelFormat format) {
        for (var repo : repositories) {
            try {
                if (repo.isCached(modelId, format))
                    return true;
            } catch (Exception e) {
                // Ignore
            }
        }
        return false;
    }

    public void evictCache(String modelId, ModelFormat format) {
        for (var repo : repositories) {
            try {
                repo.evictCache(modelId, format);
            } catch (Exception e) {
                // Ignore
            }
        }
    }

    public ModelDescriptor resolve(ModelRef ref) {
        for (var repo : repositories) {
            try {
                if (repo.supports(ref)) {
                    return repo.resolve(ref);
                }
            } catch (Exception e) {
                // Ignore
            }
        }
        return null;
    }

    public ModelArtifact fetch(ModelDescriptor descriptor) {
        // This is tricky because we don't know which repo the descriptor belongs to
        // But usually descriptors have URI that could help, or we just try all
        for (var repo : repositories) {
            try {
                var artifact = repo.fetch(descriptor);
                if (artifact != null)
                    return artifact;
            } catch (Exception e) {
                // Ignore
            }
        }
        return null;
    }

    public boolean supports(ModelRef ref) {
        for (var repo : repositories) {
            if (repo.supports(ref))
                return true;
        }
        return false;
    }
}