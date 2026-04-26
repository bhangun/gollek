package tech.kayys.gollek.registry.repository;

import io.quarkus.cache.CacheResult;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.Pageable;
import tech.kayys.gollek.spi.model.ModelArtifact;
import tech.kayys.gollek.spi.model.ModelDescriptor;
import tech.kayys.gollek.spi.model.ModelRef;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Cached wrapper around model repositories.
 */
import io.quarkus.cache.CacheInvalidateAll;

@ApplicationScoped
public class CachedModelRepository {

    @Inject
    jakarta.enterprise.inject.Instance<tech.kayys.gollek.model.core.ModelRepository> repositories;

    @Inject
    ModelRepositoryRegistry registry;

    /**
     * Get repositories in prioritized order: Local > HuggingFace > Others (Kaggle, etc.)
     */
    private List<tech.kayys.gollek.model.core.ModelRepository> getPrioritizedRepositories() {
        List<tech.kayys.gollek.model.core.ModelRepository> local = new ArrayList<>();
        List<tech.kayys.gollek.model.core.ModelRepository> hf = new ArrayList<>();
        List<tech.kayys.gollek.model.core.ModelRepository> others = new ArrayList<>();

        for (var repo : repositories) {
            String className = repo.getClass().getSimpleName();
            if (className.contains("LocalModelRepository")) {
                local.add(repo);
            } else if (className.contains("HuggingFaceRepository")) {
                hf.add(repo);
            } else {
                others.add(repo);
            }
        }

        List<tech.kayys.gollek.model.core.ModelRepository> prioritized = new ArrayList<>();
        prioritized.addAll(local);
        prioritized.addAll(hf);
        prioritized.addAll(others);
        return prioritized;
    }

    @CacheResult(cacheName = "model-manifests")
    public Uni<ModelManifest> findById(String modelId, String requestId) {
        // Try sequentially in prioritized order to avoid unnecessary remote calls
        Uni<ModelManifest> resultChain = Uni.createFrom().nullItem();
        
        for (var repo : getPrioritizedRepositories()) {
            resultChain = resultChain.onItem().ifNull().switchTo(() -> 
                repo.findById(modelId, requestId).onFailure().recoverWithNull()
            );
        }
        
        return resultChain;
    }

    public Uni<List<ModelManifest>> list(String requestId, Pageable pageable) {
        List<Uni<List<ModelManifest>>> repositoryLists = new ArrayList<>();
        for (var repo : getPrioritizedRepositories()) {
            repositoryLists.add(repo.list(requestId, pageable)
                    .onFailure().recoverWithItem(List.of()));
        }

        return Uni.combine().all().unis(repositoryLists).with(results -> {
            Set<String> seenIds = new HashSet<>();
            List<ModelManifest> all = new ArrayList<>();
            for (Object result : results) {
                @SuppressWarnings("unchecked")
                List<ModelManifest> list = (List<ModelManifest>) result;
                for (ModelManifest m : list) {
                    String uniqueKey = m.modelId() + (m.artifacts() != null ? m.artifacts().keySet().toString() : "");
                    if (seenIds.add(uniqueKey)) {
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

    @CacheInvalidateAll(cacheName = "model-manifests")
    public void invalidateCache() {
        // Clear caches (used by SDK when pulling models)
    }

    public Uni<Void> delete(String modelId, String requestId) {
        List<Uni<Void>> deletions = new ArrayList<>();
        for (var repo : getPrioritizedRepositories()) {
            deletions.add(repo.delete(modelId, requestId).onFailure().recoverWithNull());
        }

        return Uni.join().all(deletions).andCollectFailures().replaceWithVoid();
    }

    public Path downloadArtifact(ModelManifest manifest, ModelFormat format)
            throws tech.kayys.gollek.model.exception.ArtifactDownloadException {
        for (var repo : getPrioritizedRepositories()) {
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
        for (var repo : getPrioritizedRepositories()) {
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
        for (var repo : getPrioritizedRepositories()) {
            try {
                repo.evictCache(modelId, format);
            } catch (Exception e) {
                // Ignore
            }
        }
    }

    public ModelDescriptor resolve(ModelRef ref) {
        for (var repo : getPrioritizedRepositories()) {
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
        for (var repo : getPrioritizedRepositories()) {
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
        for (var repo : getPrioritizedRepositories()) {
            if (repo.supports(ref))
                return true;
        }
        return false;
    }
}
