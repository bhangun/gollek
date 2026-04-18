package tech.kayys.gollek.inference.llamacpp;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;
import jakarta.enterprise.inject.Disposes;
import jakarta.inject.Inject;

@ApplicationScoped
public class LlamaCppBeanProducer {

    @Inject
    LlamaCppProviderConfig config;

    @Produces
    @ApplicationScoped
    public LlamaCppBinding llamaCppBinding() {
        if (!config.enabled()) {
            return null;
        }
        try {
            return LlamaCppBinding.load(config);
        } catch (Exception e) {
            // Log and return null to prevent startup crash
            return null;
        }
    }

    public void dispose(@Disposes LlamaCppBinding binding) {
        // Provider shutdown owns native backend lifecycle.
        // Do not call backendFree() here because CDI destroy order can trigger
        // native cleanup before all GGUF sessions/runners are fully closed.
    }
}
