package tech.kayys.golek.runtime;

import io.quarkus.runtime.ShutdownEvent;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

/**
 * Graceful shutdown handler
 */
@ApplicationScoped
public class ShutdownHandler {

    private static final Logger LOG = Logger.getLogger(ShutdownHandler.class);

    @Inject
    InferenceEngine engine;

    @Inject
    ModelRunnerFactory runnerFactory;

    void onShutdown(@Observes ShutdownEvent event) {
        LOG.info("Shutting down Wayang Inference Platform...");

        // Shutdown engine
        engine.shutdown();

        // Close all runners
        runnerFactory.closeAll();

        LOG.info("Shutdown complete");
    }
}