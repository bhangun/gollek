package tech.kayys.golek.model.repository.hf;

import tech.kayys.golek.model.core.ModelRepository;
import tech.kayys.golek.model.core.ModelRepositoryProvider;
import tech.kayys.golek.model.core.RepositoryContext;

public final class HuggingFaceRepositoryProvider implements ModelRepositoryProvider {

    @Override
    public String scheme() {
        return "hf";
    }

    @Override
    public ModelRepository create(RepositoryContext context) {
        return new HuggingFaceRepository(context.cacheDir());
    }
}