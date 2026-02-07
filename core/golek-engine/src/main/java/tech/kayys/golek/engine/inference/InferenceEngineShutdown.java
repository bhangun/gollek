package tech.kayys.golek.engine.inference;

import io.quarkus.runtime.ShutdownEvent;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import tech.kayys.golek.spi.context.EngineContext;
import tech.kayys.golek.spi.plugin.PluginRegistry;
import tech.kayys.golek.engine.plugin.PluginLoader;

import org.jboss.logging.Logger;

import java.time.Duration;
import java.time.Instant;

/**
 * Handles graceful shutdown of the inference engine.
 */
@ApplicationScoped
public class InferenceEngineShutdown {

    private static final Logger LOG = Logger.getLogger(InferenceEngineShutdown.class);

    @Inject
    PluginLoader pluginLoader;

    @Inject
    PluginRegistry pluginRegistry;

    @Inject
    EngineContext engineContext;

    /**
     * Handle shutdown event
     */
    void onShutdown(@Observes ShutdownEvent event) {
        LOG.info("========================================");
        LOG.info("Shutting down Wayang Inference Engine...");
        LOG.info("========================================");

        Instant start = Instant.now();

        try {
            shutdown().await().atMost(Duration.ofSeconds(30));

            Duration elapsed = Duration.between(start, Instant.now());
            LOG.infof("✓ Inference engine shutdown completed in %d ms", elapsed.toMillis());
        } catch (Exception e) {
            LOG.error("✗ Error during shutdown", e);
        } finally {
            LOG.info("========================================");
        }
    }

    /**
     * Shutdown sequence
     */
    private Uni<Void> shutdown() {
        return Uni.createFrom().voidItem()
                // Step 1: Shutdown plugins
                .onItem().transformToUni(v -> {
                    LOG.info("Step 1/2: Shutting down plugins...");
                    return pluginLoader.shutdownAll()
                            .onItem()
                            .invoke(() -> LOG.infof("  → %d plugins shutdown", pluginRegistry.all().size()));
                })

                // Step 2: Cleanup (No-op if not available in SPI)
                .onItem().transformToUni(v -> {
                    LOG.info("Step 2/2: Finishing shutdown...");
                    return Uni.createFrom().voidItem()
                            .onItem().invoke(() -> LOG.info("  → Engine context shutdown signal processed"));
                });
    }
}