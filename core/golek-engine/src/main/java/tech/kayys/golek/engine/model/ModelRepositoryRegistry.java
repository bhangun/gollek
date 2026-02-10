package tech.kayys.golek.engine.model;

import java.util.*;

import tech.kayys.golek.spi.model.ModelArtifact;
import tech.kayys.golek.spi.model.ModelRef;
import tech.kayys.golek.model.core.ModelRepository;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;

@ApplicationScoped
public class ModelRepositoryRegistry {

    private final Map<String, ModelRepository> repositories = new HashMap<>();

    @Inject
    public ModelRepositoryRegistry(Instance<ModelRepository> repos) {
        // Collect available repositories by scheme?
        // Actually ModelRepository doesn't expose scheme directly in the interface I
        // saw.
        // Assuming we need a way to distinguish them.
        // For now, let's just inject what's available.
        for (var repo : repos) {
            // TODO: How to get scheme? SPI might need update or we rely on implementation
            // class.
            // implementation might be "LocalModelRepository", "S3ModelRepository" etc.
            // Checking if ModelRepository has scheme() method.
            // Based on previous view, it DOES NOT.
            // This is a problem.
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