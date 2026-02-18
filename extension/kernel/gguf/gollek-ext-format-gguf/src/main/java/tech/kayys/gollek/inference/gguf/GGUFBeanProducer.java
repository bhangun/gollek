package tech.kayys.gollek.inference.gguf;

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
        // Provider shutdown owns native backend lifecycle.
        // Do not call backendFree() here because CDI destroy order can trigger
        // native cleanup before all GGUF sessions/runners are fully closed.
    }
}
