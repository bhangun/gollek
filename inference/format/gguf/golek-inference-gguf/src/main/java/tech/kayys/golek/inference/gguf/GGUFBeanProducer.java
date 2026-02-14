package tech.kayys.golek.inference.gguf;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;
import jakarta.enterprise.inject.Disposes;
import jakarta.inject.Inject;

@ApplicationScoped
public class GGUFBeanProducer {

    @Inject
    GGUFProviderConfig config;

    @Produces
    @ApplicationScoped
    public LlamaCppBinding llamaCppBinding() {
        if (!config.enabled()) {
            return null;
        }
        return LlamaCppBinding.load(config);
    }

    public void dispose(@Disposes LlamaCppBinding binding) {
        if (binding != null) {
            binding.backendFree();
        }
    }
}
