package tech.kayys.gollek.engine.inference;

import io.quarkus.runtime.ShutdownEvent;
import io.quarkus.runtime.Startup;
import io.quarkus.runtime.StartupEvent;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.plugin.PluginContext;
import tech.kayys.gollek.spi.plugin.PluginRegistry;
import tech.kayys.gollek.spi.plugin.PluginHealth;
import tech.kayys.gollek.spi.inference.InferenceEngine;
import tech.kayys.gollek.engine.context.DefaultEngineContext;
import tech.kayys.gollek.plugin.core.PluginLoader;

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

    @ConfigProperty(name = "gollek.inference.engine.startup.timeout", defaultValue = "30s")
    Duration startupTimeout;

    @ConfigProperty(name = "gollek.inference.engine.enabled", defaultValue = "true")
    boolean engineEnabled;

    @ConfigProperty(name = "gollek.inference.engine.startup.fail-on-plugin-error", defaultValue = "false")
    boolean failOnPluginError;

    @ConfigProperty(name = "gollek.inference.engine.startup.min-plugins", defaultValue = "0")
    int minPluginsRequired;

    private volatile boolean initialized = false;
    private Instant startupTime;
    private final Map<String, Duration> phaseTimings = new LinkedHashMap<>();
    private final AtomicInteger successfulPlugins = new AtomicInteger(0);
    private final AtomicInteger failedPlugins = new AtomicInteger(0);

    /**
     * Bootstrap on application startup
     */
    public void initialize() {
        if (!engineEnabled) {
            LOG.warn("Inference engine is disabled via configuration");
            return;
        }

        LOG.info("========================================");
        LOG.info("🚀 Starting Gollek Inference Engine...");
        LOG.info("========================================");

        Instant start = Instant.now();
        this.startupTime = start;

        try {
            validateConfiguration();
            
            bootstrap().await().atMost(startupTimeout);

            Duration elapsed = Duration.between(start, Instant.now());
            LOG.infof("✅ Inference engine started successfully in %d ms", elapsed.toMillis());

            collectMetrics();
            LOG.info("========================================");

            initialized = true;
        } catch (Throwable e) {
            LOG.error("========================================");
            LOG.error("❌ Failed to start inference engine", e);
            LOG.error("========================================");
            throw new RuntimeException("Inference engine startup failed", e);
        }
    }

    /**
     * Graceful shutdown handling
     */
    @jakarta.annotation.PreDestroy
    public void shutdown() {
        if (!initialized) {
            return;
        }

        LOG.info("========================================");
        LOG.info("🛑 Shutting down Gollek Inference Engine...");
        LOG.info("========================================");

        try {
            if (engineContext instanceof DefaultEngineContext defaultContext) {
                defaultContext.setRunning(false);
            }

            pluginLoader.shutdownAll().await().atMost(Duration.ofSeconds(10));
            
            LOG.info("✅ Inference engine shutdown complete");
            LOG.info("========================================");
        } catch (Exception e) {
            LOG.error("❌ Error during inference engine shutdown", e);
        } finally {
            initialized = false;
        }
    }

    /**
     * Main bootstrap sequence
     */
    private Uni<Void> bootstrap() {
        return Uni.createFrom().voidItem()
                .onItem().transformToUni(v -> {
                    Instant phaseStart = Instant.now();
                    LOG.info("[1/4] Initializing engine context...");
                    if (engineContext instanceof DefaultEngineContext defaultContext) {
                        defaultContext.setRunning(true);
                    }
                    phaseTimings.put("context_init", Duration.between(phaseStart, Instant.now()));
                    return Uni.createFrom().voidItem();
                })
                .onItem().transformToUni(v -> {
                    Instant phaseStart = Instant.now();
                    LOG.info("[2/4] Loading plugins...");
                    return pluginLoader.loadAll()
                            .onItem().invoke(count -> {
                                LOG.infof("  → Loaded %d plugins", count);
                                phaseTimings.put("plugin_load", Duration.between(phaseStart, Instant.now()));
                            })
                            .replaceWithVoid();
                })
                .onItem().transformToUni(v -> {
                    Instant phaseStart = Instant.now();
                    LOG.info("[3/4] Initializing plugins...");
                    return initializePlugins()
                            .onItem().invoke(() -> {
                                phaseTimings.put("plugin_init", Duration.between(phaseStart, Instant.now()));
                            });
                })
                .onItem().transformToUni(v -> {
                    Instant phaseStart = Instant.now();
                    LOG.info("[4/4] Verifying engine health...");
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
        if (failOnPluginError) {
            return pluginLoader.initializeAll(createSystemContext())
                    .onItem().invoke(() -> {
                        int total = pluginRegistry.all().size();
                        successfulPlugins.set(total);
                        LOG.infof("  → Initialized %d plugins", total);
                    });
        } else {
            return initializePluginsGracefully(createSystemContext());
        }
    }

    private PluginContext createSystemContext() {
        return new PluginContext() {
            @Override
            public String getPluginId() {
                return "system";
            }

            @Override
            public Optional<String> getConfig(String key) {
                return Optional.empty();
            }
        };
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
                            LOG.errorf(error, "❌ Failed to initialize plugin: %s", plugin.id());
                            return null;
                        }))
                .toList();

        return Uni.join().all(initializations).andCollectFailures()
                .replaceWithVoid()
                .onItem().invoke(() -> {
                    int successCount = successfulPlugins.get();
                    int failCount = failedPlugins.get();

                    if (failCount > 0) {
                        LOG.warnf("  → Initialized %d/%d plugins (%d failed)",
                                successCount, successCount + failCount, failCount);
                    } else {
                        LOG.infof("  → Initialized %d plugins", successCount);
                    }

                    if (successCount < minPluginsRequired) {
                        throw new RuntimeException(
                                String.format("Insufficient plugins initialized: %d < %d required",
                                        successCount, minPluginsRequired));
                    }
                });
    }

    /**
     * Verify engine and plugin health
     */
    private Uni<Void> verifyHealth() {
        return Uni.createFrom().item(() -> {
            var engineHealth = engine.health();
            LOG.infof("  → Engine: %s", engineHealth.getStatus());

            var pluginHealthMap = pluginLoader.checkAllHealth();
            long healthyCount = pluginHealthMap.values().stream()
                    .filter(PluginHealth::isHealthy)
                    .count();

            LOG.infof("  → Plugins: %d/%d healthy", healthyCount, pluginHealthMap.size());

            pluginHealthMap.forEach((id, health) -> {
                if (!health.isHealthy()) {
                    LOG.warnf("  ⚠ Plugin '%s' is %s: %s",
                            id, health.status(), health.message());
                }
            });

            return null;
        });
    }

    public boolean isInitialized() {
        return initialized;
    }

    public Instant getStartupTime() {
        return startupTime;
    }

    public Duration getUptime() {
        return startupTime != null ? Duration.between(startupTime, Instant.now()) : Duration.ZERO;
    }

    private void validateConfiguration() {
        if (startupTimeout.isNegative() || startupTimeout.isZero()) {
            throw new IllegalArgumentException("Startup timeout must be positive: " + startupTimeout);
        }
        if (minPluginsRequired < 0) {
            throw new IllegalArgumentException("Minimum plugins required cannot be negative: " + minPluginsRequired);
        }
    }

    private void collectMetrics() {
        LOG.info("Startup Metrics:");
        LOG.infof("  Total Time: %d ms", Duration.between(startupTime, Instant.now()).toMillis());

        phaseTimings.forEach((phase, duration) -> {
            LOG.infof("    %s: %d ms", phase, duration.toMillis());
        });

        LOG.infof("  Plugins: %d successful, %d failed",
                successfulPlugins.get(), failedPlugins.get());
    }

    public Map<String, Object> getPluginStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("total", pluginRegistry.all().size());
        stats.put("successful", successfulPlugins.get());
        stats.put("failed", failedPlugins.get());
        stats.put("uptime", getUptime().toString());
        return stats;
    }
}