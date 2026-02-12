package tech.kayys.golek.engine.inference;

import io.quarkus.runtime.Startup;
import io.quarkus.runtime.StartupEvent;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import tech.kayys.golek.spi.context.EngineContext;
import tech.kayys.golek.spi.plugin.PluginContext;
import tech.kayys.golek.spi.plugin.PluginRegistry;
import tech.kayys.golek.spi.plugin.PluginHealth;
import tech.kayys.golek.core.inference.InferenceEngine;
import tech.kayys.golek.engine.plugin.PluginLoader;
import tech.kayys.golek.engine.context.DefaultEngineContext;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.time.Duration;
import java.time.Instant;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Bootstrap class that initializes the inference engine on startup.
 * Responsible for:
 * - Loading and initializing all plugins
 * - Setting up engine context
 * - Performing health checks
 * - Graceful shutdown handling
 */
@Startup
@ApplicationScoped
public class InferenceEngineBootstrap {

    private static final Logger LOG = Logger.getLogger(InferenceEngineBootstrap.class);

    @Inject
    InferenceEngine engine;

    @Inject
    PluginLoader pluginLoader;

    @Inject
    PluginRegistry pluginRegistry;

    @Inject
    EngineContext engineContext;

    @ConfigProperty(name = "wayang.inference.engine.startup.timeout", defaultValue = "30s")
    Duration startupTimeout;

    @ConfigProperty(name = "wayang.inference.engine.enabled", defaultValue = "true")
    boolean engineEnabled;

    @ConfigProperty(name = "wayang.inference.engine.startup.fail-on-plugin-error", defaultValue = "false")
    boolean failOnPluginError;

    @ConfigProperty(name = "wayang.inference.engine.startup.min-plugins", defaultValue = "0")
    int minPluginsRequired;

    private volatile boolean initialized = false;
    private Instant startupTime;
    private final Map<String, Duration> phaseTimings = new LinkedHashMap<>();
    private final AtomicInteger successfulPlugins = new AtomicInteger(0);
    private final AtomicInteger failedPlugins = new AtomicInteger(0);

    /**
     * Bootstrap on application startup
     */
    void onStart(@Observes StartupEvent event) {

        if (!engineEnabled) {

            LOG.warn("Inference engine is disabled");
            return;
        }

        LOG.info("========================================");
        LOG.info("Starting Wayang Inference Engine...");
        LOG.info("========================================");

        Instant start = Instant.now();
        this.startupTime = start;

        try {
            // Validate configuration first

            validateConfiguration();

            bootstrap().await().atMost(startupTimeout);

            Duration elapsed = Duration.between(start, Instant.now());

            // LOG.infof("✓ Inference engine started successfully in %d ms",
            // elapsed.toMillis());

            // Log metrics

            collectMetrics();

            LOG.info("========================================");

            initialized = true;

        } catch (Throwable e) {

            e.printStackTrace();
            // LOG.error("========================================");
            // LOG.error("✗ Failed to start inference engine", e);
            // LOG.error("========================================");
            throw new RuntimeException("Inference engine startup failed", e);
        }
    }

    /**
     * Main bootstrap sequence
     */
    private Uni<Void> bootstrap() {
        return Uni.createFrom().voidItem()
                // Step 1: Initialize engine context
                .onItem().transformToUni(v -> {
                    Instant phaseStart = Instant.now();
                    LOG.info("Step 1/4: Initializing engine context...");
                    if (engineContext instanceof DefaultEngineContext defaultContext) {
                        defaultContext.setRunning(true);
                    }
                    phaseTimings.put("context_init", Duration.between(phaseStart, Instant.now()));
                    return Uni.createFrom().voidItem();
                })

                // Step 2: Load plugins
                .onItem().transformToUni(v -> {
                    Instant phaseStart = Instant.now();
                    LOG.info("Step 2/4: Loading plugins...");
                    return pluginLoader.loadAll()
                            .onItem().invoke(count -> {
                                LOG.infof("  → Loaded %d plugins", count);
                                phaseTimings.put("plugin_load", Duration.between(phaseStart, Instant.now()));
                            })
                            .replaceWithVoid();
                })

                // Step 3: Initialize plugins
                .onItem().transformToUni(v -> {
                    Instant phaseStart = Instant.now();
                    LOG.info("Step 3/4: Initializing plugins...");
                    return initializePlugins()
                            .onItem().invoke(() -> {
                                phaseTimings.put("plugin_init", Duration.between(phaseStart, Instant.now()));
                            });
                })

                // Step 4: Verify engine health
                .onItem().transformToUni(v -> {
                    Instant phaseStart = Instant.now();
                    LOG.info("Step 4/4: Verifying engine health...");
                    return verifyHealth()
                            .onItem().invoke(() -> {
                                phaseTimings.put("health_check", Duration.between(phaseStart, Instant.now()));
                            });
                });
    }

    /**
     * Initialize all plugins with proper context
     */
    private Uni<Void> initializePlugins() {
        // Create a default plugin context for bootstrap
        PluginContext context = new PluginContext() {
            @Override
            public String getPluginId() {
                return "system";
            }

            @Override
            public Optional<String> getConfig(String key) {
                return Optional.empty();
            }
        };

        if (failOnPluginError) {
            // Fail fast on any plugin initialization
            return pluginLoader.initializeAll(context)
                    .onItem().invoke(() -> {
                        int total = pluginRegistry.all().size();
                        successfulPlugins.set(total);
                        LOG.infof("  → Initialized %d plugins", total);
                        // logPluginSummary(); // Temporarily disabled until generic stats implemented
                    });
        } else {
            // Graceful degradation: continue even if some plugins fail
            return initializePluginsGracefully(context);
        }
    }

    /**
     * Initialize plugins with graceful degradation
     */
    private Uni<Void> initializePluginsGracefully(PluginContext context) {
        List<Uni<Void>> initializations = pluginRegistry.all().stream()
                .map(plugin -> Uni.createFrom().voidItem()
                        .onItem().invoke(() -> plugin.initialize(context))
                        .onItem().invoke(() -> {
                            successfulPlugins.incrementAndGet();
                            LOG.debugf("✓ Initialized plugin: %s", plugin.id());
                        })
                        .onFailure().recoverWithItem(error -> {
                            failedPlugins.incrementAndGet();
                            handlePluginInitializationFailure(plugin.id(), error);
                            return null;
                        }))
                .toList();

        return Uni.join().all(initializations).andCollectFailures()
                .replaceWithVoid()
                .onItem().invoke(() -> {
                    int total = successfulPlugins.get();
                    int failed = failedPlugins.get();

                    if (failed > 0) {
                        LOG.warnf("  → Initialized %d/%d plugins (%d failed)",
                                total, total + failed, failed);
                    } else {
                        LOG.infof("  → Initialized %d plugins", total);
                    }

                    // Verify minimum plugins requirement
                    if (total < minPluginsRequired) {
                        throw new RuntimeException(
                                String.format("Insufficient plugins initialized: %d < %d required",
                                        total, minPluginsRequired));
                    }
                });
    }

    /**
     * Handle plugin initialization failure
     */
    private void handlePluginInitializationFailure(String pluginId, Throwable error) {
        LOG.errorf(error, "✗ Failed to initialize plugin: %s", pluginId);
        LOG.warnf("  → Engine will continue without plugin: %s", pluginId);
    }

    /**
     * Verify engine and plugin health
     */
    private Uni<Void> verifyHealth() {
        return Uni.createFrom().item(() -> {
            // Check engine health
            var engineHealth = engine.health();
            LOG.infof("  → Engine: %s", engineHealth.getStatus());

            // Check plugin health
            var pluginHealthMap = pluginLoader.checkAllHealth();
            long healthyCount = pluginHealthMap.values().stream()
                    .filter(PluginHealth::isHealthy)
                    .count();

            LOG.infof("  → Plugins: %d/%d healthy", healthyCount, pluginHealthMap.size());

            // Log unhealthy plugins
            pluginHealthMap.forEach((id, health) -> {
                if (!health.isHealthy()) {
                    LOG.warnf("  ⚠ Plugin '%s' is %s: %s",
                            id, health.status(), health.message());
                }
            });

            return null;
        });
    }

    /**
     * Log plugin summary by phase
     */
    private void logPluginSummary() {
        // This method is temporarily disabled as phase-bound statistics are not yet
        // generic.
        // var stats = ((DefaultPluginRegistry) pluginRegistry).getStatistics();

        // LOG.info(" Plugin Summary:");
        // LOG.infof(" Total: %d", stats.totalPlugins());
        // LOG.infof(" Phase-bound: %d", stats.phasePlugins());

        // stats.pluginsPerPhase().forEach((phase, count) -> {
        // if (count > 0) {
        // LOG.infof(" %s: %d", phase.getDisplayName(), count);
        // }
        // });
    }

    /**
     * Check if engine is initialized
     */
    public boolean isInitialized() {
        return initialized;
    }

    /**
     * Get startup time
     */
    public Instant getStartupTime() {
        return startupTime;
    }

    /**
     * Get uptime duration
     */
    public Duration getUptime() {
        if (startupTime == null) {
            return Duration.ZERO;
        }
        return Duration.between(startupTime, Instant.now());
    }

    /**
     * Validate configuration before startup
     */
    private void validateConfiguration() {
        LOG.debug("Validating configuration...");

        if (startupTimeout.isNegative() || startupTimeout.isZero()) {
            throw new IllegalArgumentException(
                    "Startup timeout must be positive: " + startupTimeout);
        }

        if (minPluginsRequired < 0) {
            throw new IllegalArgumentException(
                    "Minimum plugins required cannot be negative: " + minPluginsRequired);
        }

        LOG.debugf("Configuration valid: timeout=%s, minPlugins=%d, failOnError=%s",
                startupTimeout, minPluginsRequired, failOnPluginError);
    }

    /**
     * Collect and log startup metrics
     */
    private void collectMetrics() {
        LOG.info("Startup Metrics:");
        LOG.infof("  Total Time: %d ms", Duration.between(startupTime, Instant.now()).toMillis());

        phaseTimings.forEach((phase, duration) -> {
            LOG.infof("    %s: %d ms", phase, duration.toMillis());
        });

        LOG.infof("  Plugins: %d successful, %d failed",
                successfulPlugins.get(), failedPlugins.get());
    }

    /**
     * Get detailed plugin statistics
     */
    public Map<String, Object> getPluginStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("total", pluginRegistry.all().size());
        stats.put("successful", successfulPlugins.get());
        stats.put("failed", failedPlugins.get());

        return stats;
    }
}