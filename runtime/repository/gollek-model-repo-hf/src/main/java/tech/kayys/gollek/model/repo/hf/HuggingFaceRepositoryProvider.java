package tech.kayys.gollek.model.repo.hf;


import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.alkhawarizm.spi.model.ModelRepository;
import tech.kayys.alkhawarizm.spi.model.ModelRepositoryProvider;
import tech.kayys.alkhawarizm.spi.model.RepositoryContext;

@ApplicationScoped
public final class HuggingFaceRepositoryProvider implements ModelRepositoryProvider {

    @Inject
    HuggingFaceClient client;

    @Inject
    HuggingFaceConfig config;

    @Inject
    tech.kayys.gollek.model.repo.local.ManifestStore manifestStore;

    @Override
    public String scheme() {
        return "hf";
    }

    public ModelRepository create(RepositoryContext context) {
        return new HuggingFaceRepository(context.cacheDir(), client, config, manifestStore);
    }
}