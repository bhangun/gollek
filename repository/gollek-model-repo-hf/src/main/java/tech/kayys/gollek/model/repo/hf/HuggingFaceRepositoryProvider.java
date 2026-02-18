package tech.kayys.gollek.model.repo.hf;


import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.model.core.ModelRepository;
import tech.kayys.gollek.model.core.ModelRepositoryProvider;
import tech.kayys.gollek.model.core.RepositoryContext;

@ApplicationScoped
public final class HuggingFaceRepositoryProvider implements ModelRepositoryProvider {

    @Inject
    HuggingFaceClient client;

    @Override
    public String scheme() {
        return "hf";
    }

    @Override
    public ModelRepository create(RepositoryContext context) {
        return new HuggingFaceRepository(context.cacheDir(), client);
    }
}