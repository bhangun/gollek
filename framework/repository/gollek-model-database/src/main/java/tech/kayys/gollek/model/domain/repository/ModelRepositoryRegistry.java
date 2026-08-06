package tech.kayys.gollek.registry.repository;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import tech.kayys.alkhawarizm.spi.model.ModelArtifact;
import tech.kayys.alkhawarizm.spi.model.ModelRef;
import tech.kayys.alkhawarizm.spi.model.ModelRepository;

import java.util.HashMap;
import java.util.Map;

/**
 * Registry for model repository implementations.
 */
@ApplicationScoped
public class ModelRepositoryRegistry {

    private final Map<String, ModelRepository> repositories = new HashMap<>();

    @Inject
    public ModelRepositoryRegistry(Instance<ModelRepository> repos) {
        // Repositories are registered dynamically via CDI
        for (var repo : repos) {
            // Repository registration logic
            // Implementation depends on how repositories expose their scheme
        }
    }

    public ModelArtifact load(ModelRef ref) {
        var repo = repositories.get(ref.scheme());
        if (repo == null || !repo.supports(ref)) {
            throw new IllegalArgumentException("Unsupported repository scheme: " + ref.scheme());
        }
        var descriptor = repo.resolve(ref);
        return repo.fetch(descriptor);
    }
}
