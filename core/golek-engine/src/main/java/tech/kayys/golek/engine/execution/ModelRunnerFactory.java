package tech.kayys.golek.engine.execution;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import lombok.extern.slf4j.Slf4j;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.spi.exception.InferenceException;
import tech.kayys.golek.engine.model.ModelRunner;
import tech.kayys.golek.engine.model.RunnerConfiguration;

import org.eclipse.microprofile.config.inject.ConfigProperty;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Factory for creating and managing ModelRunner instances.
 * 
 * <p>
 * Features:
 * <ul>
 * <li>Runner instance pooling (warm pool)</li>
 * <li>Lazy initialization</li>
 * <li>LRU eviction when pool is full</li>
 * <li>Health monitoring</li>
 * <li>CDI-based runner discovery</li>
 * </ul>
 * 
 * @author bhangun
 * @since 1.0.0
 */
@ApplicationScoped
@Slf4j
public class ModelRunnerFactory {

    // Inject all available runner implementations via CDI
    @Inject
    Instance<ModelRunner> runnerProviders;

    @ConfigProperty(name = "inference.warm-pool.enabled", defaultValue = "true")
    boolean warmPoolEnabled;

    @ConfigProperty(name = "inference.warm-pool.max-size", defaultValue = "10")
    int maxPoolSize;

    // Warm pool: key = "runnerName:modelId:version"
    private final Map<String, ModelRunner> warmPool = new ConcurrentHashMap<>();

    // LRU tracker
    private final Map<String, Long> accessTimes = new ConcurrentHashMap<>();

    // Available runner names
    private final Set<String> availableRunnerNames = new HashSet<>();

    @jakarta.annotation.PostConstruct
    void init() {
        // Discover available runners from CDI
        for (ModelRunner runner : runnerProviders) {
            String runnerName = runner.name();
            availableRunnerNames.add(runnerName);
            log.info("Discovered runner: " + runnerName);
        }

        log.info("ModelRunnerFactory initialized with " + availableRunnerNames.size() + " runners");
    }

    /**
     * Get or create runner instance.
     * 
     * @param runnerName Name of the runner (e.g., "litert-cpu")
     * @param manifest   Model manifest
     * @return Initialized runner instance
     */
    public ModelRunner getOrCreateRunner(String runnerName, ModelManifest manifest) {
        String poolKey = buildPoolKey(runnerName, manifest);

        // Check warm pool first
        if (warmPoolEnabled) {
            ModelRunner cachedRunner = warmPool.get(poolKey);
            if (cachedRunner != null) {
                accessTimes.put(poolKey, System.currentTimeMillis());
                log.debug("Returning cached runner: " + poolKey);
                return cachedRunner;
            }
        }

        // Create new runner
        ModelRunner runner = createRunner(runnerName, manifest);

        // Add to warm pool
        if (warmPoolEnabled) {
            addToPool(poolKey, runner);
        }

        return runner;
    }

    /**
     * Get runner by name (without model).
     */
    public ModelRunner getRunner(String runnerName) {
        // Find any runner with this name in pool
        return warmPool.values().stream()
                .filter(r -> r.name().equals(runnerName))
                .findFirst()
                .orElse(null);
    }

    /**
     * Create new runner instance.
     */
    private ModelRunner createRunner(String runnerName, ModelManifest manifest) {
        log.info("Creating runner: runnerName=" + runnerName + ", model=" + manifest.name());

        // Find runner provider
        ModelRunner runnerTemplate = null;
        for (ModelRunner runner : runnerProviders) {
            if (runner.name().equals(runnerName)) {
                runnerTemplate = runner;
                break;
            }
        }

        if (runnerTemplate == null) {
            throw new InferenceException("Runner not found: " + runnerName);
        }

        try {
            // Create new instance (CDI-managed)
            ModelRunner newRunner = runnerTemplate.getClass().getDeclaredConstructor().newInstance();

            // Initialize with model
            RunnerConfiguration config = buildRunnerConfig(runnerName);
            newRunner.initialize(manifest, config);

            log.info("Runner created and initialized: " + runnerName);
            return newRunner;

        } catch (Exception e) {
            log.error("Failed to create runner: " + runnerName, e);
            throw new InferenceException("Failed to create runner: " + runnerName, e);
        }
    }

    /**
     * Add runner to warm pool with LRU eviction.
     */
    private void addToPool(String poolKey, ModelRunner runner) {
        // Check pool size
        if (warmPool.size() >= maxPoolSize) {
            evictLRU();
        }

        warmPool.put(poolKey, runner);
        accessTimes.put(poolKey, System.currentTimeMillis());

        log.debug("Runner added to pool: " + poolKey + " (pool size: " + warmPool.size() + ")");
    }

    /**
     * Evict least recently used runner.
     */
    private void evictLRU() {
        if (warmPool.isEmpty()) {
            return;
        }

        // Find LRU entry
        String lruKey = accessTimes.entrySet().stream()
                .min(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(null);

        if (lruKey != null) {
            ModelRunner evicted = warmPool.remove(lruKey);
            accessTimes.remove(lruKey);

            // Cleanup runner
            if (evicted != null) {
                try {
                    evicted.close();
                    log.info("Evicted LRU runner: " + lruKey);
                } catch (Exception e) {
                    log.warn("Error closing evicted runner: " + lruKey, e);
                }
            }
        }
    }

    /**
     * Build pool key for runner+model combination.
     */
    private String buildPoolKey(String runnerName, ModelManifest manifest) {
        return String.format("%s:%s:%s",
                runnerName,
                manifest.name(),
                manifest.version());
    }

    /**
     * Build runner configuration from application config.
     */
    private RunnerConfiguration buildRunnerConfig(String runnerName) {
        // In production, load from ConfigProvider based on runner name
        // For now, return empty config
        return RunnerConfiguration.builder()
                .build();
    }

    /**
     * Get list of available runner names.
     */
    public List<String> getAvailableRunners() {
        return new ArrayList<>(availableRunnerNames);
    }

    /**
     * Prewarm runners for specific models.
     */
    public void prewarm(List<String> modelIds, List<String> runnerNames) {
        log.info("Prewarming runners: models=" + modelIds + ", runners=" + runnerNames);

        for (String modelId : modelIds) {
            for (String runnerName : runnerNames) {
                try {
                    // This would need actual ModelManifest
                    log.info("Would prewarm: runner=" + runnerName + ", model=" + modelId);
                    // ModelManifest manifest = loadManifest(modelId);
                    // getOrCreateRunner(runnerName, manifest);
                } catch (Exception e) {
                    log.warn("Failed to prewarm runner " + runnerName + " for model " + modelId, e);
                }
            }
        }
    }

    /**
     * Clear warm pool (for testing or maintenance).
     */
    public void clearPool() {
        log.info("Clearing warm pool (" + warmPool.size() + " runners)");

        for (Map.Entry<String, ModelRunner> entry : warmPool.entrySet()) {
            try {
                entry.getValue().close();
            } catch (Exception e) {
                log.warn("Error closing runner during pool clear: " + entry.getKey(), e);
            }
        }

        warmPool.clear();
        accessTimes.clear();
    }

    /**
     * Get pool statistics.
     */
    public PoolStats getPoolStats() {
        return new PoolStats(
                warmPool.size(),
                maxPoolSize,
                (double) warmPool.size() / maxPoolSize,
                availableRunnerNames.size());
    }

    public record PoolStats(
            int currentSize,
            int maxSize,
            double utilizationPercent,
            int totalAvailableRunners) {
    }

    @jakarta.annotation.PreDestroy
    void cleanup() {
        log.info("Shutting down ModelRunnerFactory");
        clearPool();
    }
}