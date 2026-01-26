package tech.kayys.golek.model.repository.local;

import tech.kayys.golek.model.core.ModelRepository;
import tech.kayys.golek.model.core.ModelRepositoryProvider;
import tech.kayys.golek.model.core.RepositoryContext;

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
