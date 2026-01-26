package tech.kayys.golek.runtime;

import io.quarkus.runtime.StartupEvent;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

/**
 * Application startup initialization
 */
@ApplicationScoped
public class StartupInitializer {

    private static final Logger LOG = Logger.getLogger(StartupInitializer.class);

    @Inject
    InferenceEngine engine;

    @Inject
    PluginRegistry pluginRegistry;

    @Inject
    ModelRunnerFactory runnerFactory;

    void onStart(@Observes StartupEvent event) {
        LOG.info("=".repeat(80));
        LOG.info("Wayang Inference Platform Runtime Starting...");
        LOG.info("=".repeat(80));

        // Initialize engine
        LOG.info("Initializing inference engine...");
        engine.initialize();

        // Load plugins
        int pluginCount = pluginRegistry.all().size();
        LOG.infof("Loaded %d plugins", pluginCount);

        // Warmup critical models (if configured)
        LOG.info("Warmup phase complete");

        LOG.info("=".repeat(80));
        LOG.info("Wayang Inference Platform Runtime Ready!");
        LOG.info("=".repeat(80));
    }
}