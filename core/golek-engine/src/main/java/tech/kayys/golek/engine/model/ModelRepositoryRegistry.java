package tech.kayys.golek.engine.model;

import java.util.*;

import tech.kayys.golek.engine.model.ModelArtifact;
import tech.kayys.golek.engine.model.ModelRef;
import tech.kayys.golek.model.core.RepositoryContext;

public final class ModelRepositoryRegistry {

    private final Map<String, ModelRepository> repositories = new HashMap<>();

    public ModelRepositoryRegistry(
            Collection<ModelRepositoryProvider> providers,
            RepositoryContext context) {
        for (var p : providers) {
            repositories.put(p.scheme(), p.create(context));
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