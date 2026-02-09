package tech.kayys.golek.inference.gguf;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;
import io.quarkus.runtime.Startup;

@ApplicationScoped
public class GGUFBeanProducer {

    @Produces
    @ApplicationScoped
    @Startup
    public LlamaCppBinding llamaCppBinding() {
        return LlamaCppBinding.load();
    }
}
