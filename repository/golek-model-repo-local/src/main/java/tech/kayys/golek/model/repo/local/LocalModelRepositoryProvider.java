package tech.kayys.golek.model.repo.local;


import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.golek.model.core.ModelRepository;
import tech.kayys.golek.model.core.ModelRepositoryProvider;
import tech.kayys.golek.model.core.RepositoryContext;

@ApplicationScoped
public final class LocalModelRepositoryProvider implements ModelRepositoryProvider {

    @Override
    public String scheme() {
        return "local";
    }

    @Override
    public ModelRepository create(RepositoryContext context) {
        return new LocalModelRepository();
    }
}
