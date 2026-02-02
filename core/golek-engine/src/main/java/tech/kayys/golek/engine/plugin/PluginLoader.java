package tech.kayys.golek.engine.plugin;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import tech.kayys.golek.engine.context.EngineContext;
import tech.kayys.golek.plugin.api.GolekPlugin;
import tech.kayys.golek.plugin.api.PluginContext;

import org.jboss.logging.Logger;

import java.util.*;

/**
 * Discovers and loads plugins from various sources.
 * Supports CDI discovery, ServiceLoader, and manual registration.
 */
@ApplicationScoped
public class PluginLoader {

    private static final Logger LOG = Logger.getLogger(PluginLoader.class);

    @Inject
    Instance<GolekPlugin> cdiPlugins;

    @Inject
    PluginRegistry registry;

    @Inject
    EngineContext engineContext;

    private volatile boolean loaded = false;

    /**
     * Discover and load all plugins
     */
    public Uni<Integer> loadAll() {
        if (loaded) {
            LOG.info("Plugins already loaded");
            return Uni.createFrom().item(registry.count());
        }

        LOG.info("Loading plugins...");

        return Uni.createFrom().item(() -> {
            int count = 0;

            // Load CDI plugins
            count += loadCDIPlugins();

            // Load ServiceLoader plugins
            count += loadServiceLoaderPlugins();

            loaded = true;
            LOG.infof("Loaded %d plugins", count);

            return count;
        });
    }

    /**
     * Load plugins discovered via CDI
     */
    private int loadCDIPlugins() {
        int count = 0;

        for (GolekPlugin plugin : cdiPlugins) {
            try {
                registry.register(plugin);
                count++;
            } catch (Exception e) {
                LOG.errorf(e, "Failed to register CDI plugin: %s", plugin.id());
            }
        }

        LOG.infof("Loaded %d CDI plugins", count);
        return count;
    }

    /**
     * Load plugins discovered via ServiceLoader
     */
    private int loadServiceLoaderPlugins() {
        int count = 0;

        ServiceLoader<GolekPlugin> loader = ServiceLoader.load(GolekPlugin.class);
        for (GolekPlugin plugin : loader) {
            try {
                if (!registry.isRegistered(plugin.id())) {
                    registry.register(plugin);
                    count++;
                }
            } catch (Exception e) {
                LOG.errorf(e, "Failed to register ServiceLoader plugin: %s", plugin.id());
            }
        }

        LOG.infof("Loaded %d ServiceLoader plugins", count);
        return count;
    }

    /**
     * Initialize all registered plugins
     */
    public Uni<Void> initializeAll(PluginContext context) {
        LOG.info("Initializing all plugins...");

        List<Uni<Void>> initializations = registry.getAllPlugins().stream()
                .map(plugin -> plugin.initialize(context)
                        .onItem().invoke(() -> LOG.debugf("Initialized plugin: %s", plugin.id()))
                        .onFailure().invoke(error -> LOG.errorf(error, "Failed to initialize plugin: %s", plugin.id())))
                .toList();

        return Uni.join().all(initializations).andFailFast()
                .replaceWithVoid()
                .onItem().invoke(() -> LOG.infof("Initialized %d plugins", initializations.size()));
    }

    /**
     * Shutdown all registered plugins
     */
    public Uni<Void> shutdownAll() {
        LOG.info("Shutting down all plugins...");

        List<Uni<Void>> shutdowns = registry.getAllPlugins().stream()
                .map(plugin -> plugin.shutdown()
                        .onItem().invoke(() -> LOG.debugf("Shutdown plugin: %s", plugin.id()))
                        .onFailure().invoke(error -> LOG.errorf(error, "Failed to shutdown plugin: %s", plugin.id())))
                .toList();

        return Uni.join().all(shutdowns).andCollectFailures()
                .replaceWithVoid()
                .onItem().invoke(() -> LOG.infof("Shutdown %d plugins", shutdowns.size()));
    }

    /**
     * Check health of all plugins
     */
    public Map<String, PluginHealth> checkAllHealth() {
        Map<String, PluginHealth> healthMap = new HashMap<>();

        registry.getAllPlugins().forEach(plugin -> {
            try {
                PluginHealth health = plugin.health();
                healthMap.put(plugin.id(), health);
            } catch (Exception e) {
                LOG.errorf(e, "Health check failed for plugin: %s", plugin.id());
                healthMap.put(plugin.id(),
                        PluginHealth.unhealthy("Health check threw exception: " + e.getMessage()));
            }
        });

        return healthMap;
    }
}