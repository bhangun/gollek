package tech.kayys.golek.engine.model;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.eclipse.microprofile.config.inject.ConfigProperty;

import java.util.concurrent.ScheduledExecutorService;

import com.github.benmanes.caffeine.cache.CacheLoader;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.github.benmanes.caffeine.cache.RemovalCause;

import io.quarkus.logging.Log;
import jakarta.annotation.PreDestroy;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;

import tech.kayys.golek.api.model.ModelManifest;
import tech.kayys.golek.api.exception.InferenceException;
import tech.kayys.golek.model.exception.ModelNotFoundException;
import tech.kayys.wayang.tenant.TenantConfigurationService;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.wayang.tenant.TenantId;
import tech.kayys.golek.engine.model.ModelRepository;

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
 * Factory for creating and managing runner instances with
 * warm pool, lifecycle management, and tenant isolation.
 * 
 * @author Wayang AI Team
 * @since 1.0.0
 */

@ApplicationScoped
public class ModelRunnerFactory {

    private static final int MAX_POOL_SIZE = 10;
    private static final Duration IDLE_TIMEOUT = Duration.ofMinutes(15);
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
            log.info("Discovered runner: {}", runnerName);
        }

        log.info("ModelRunnerFactory initialized with {} runners", availableRunnerNames.size());
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
                log.debug("Returning cached runner: {}", poolKey);
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
        log.info("Creating runner: runnerName={}, model={}", runnerName, manifest.getName());

        // Find runner provider
        ModelRunner runnerTemplate = null;
        for (ModelRunner runner : runnerProviders) {
            if (runner.name().equals(runnerName)) {
                runnerTemplate = runner;
                break;
            }
        }

        if (runnerTemplate == null) {
            throw new InferenceException(
                    ErrorCode.INIT_RUNNER_FAILED,
                    "Runner not found: " + runnerName);
        }

        try {
            // Create new instance (CDI-managed)
            ModelRunner newRunner = runnerTemplate.getClass().getDeclaredConstructor().newInstance();

            // Initialize with model
            RunnerConfiguration config = buildRunnerConfig(runnerName);
            newRunner.initialize(manifest, config);

            log.info("Runner created and initialized: {}", runnerName);
            return newRunner;

        } catch (Exception e) {
            log.error("Failed to create runner: {}", runnerName, e);
            throw new RunnerInitializationException(
                    ErrorCode.INIT_RUNNER_FAILED,
                    "Failed to create runner: " + runnerName,
                    e);
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

        log.debug("Runner added to pool: {} (pool size: {})", poolKey, warmPool.size());
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
                    log.info("Evicted LRU runner: {}", lruKey);
                } catch (Exception e) {
                    log.warn("Error closing evicted runner: {}", lruKey, e);
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
                manifest.getName(),
                manifest.getVersion());
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
        log.info("Prewarming runners: models={}, runners={}", modelIds, runnerNames);

        for (String modelId : modelIds) {
            for (String runnerName : runnerNames) {
                try {
                    // This would need actual ModelManifest
                    log.info("Would prewarm: runner={}, model={}", runnerName, modelId);
                    // ModelManifest manifest = loadManifest(modelId);
                    // getOrCreateRunner(runnerName, manifest);
                } catch (Exception e) {
                    log.warn("Failed to prewarm runner {} for model {}",
                            runnerName, modelId, e);
                }
            }
        }
    }

    /**
     * Clear warm pool (for testing or maintenance).
     */
    public void clearPool() {
        log.info("Clearing warm pool ({} runners)", warmPool.size());

        for (Map.Entry<String, ModelRunner> entry : warmPool.entrySet()) {
            try {
                entry.getValue().close();
            } catch (Exception e) {
                log.warn("Error closing runner during pool clear: {}", entry.getKey(), e);
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

    @Inject
    Instance<ModelRunnerProvider> runnerProviders;

    @Inject
    ModelRepository repository;

    @Inject
    TenantConfigurationService tenantConfigService;

    // Pool: (tenantId, modelId, runnerName) -> Runner instance
    private final LoadingCache<RunnerCacheKey, ModelRunner> warmPool;

    // Track usage per runner for cleanup
    private final Map<RunnerCacheKey, Instant> lastAccess;

    private ScheduledExecutorService scheduler;

    @Inject
    public ModelRunnerFactory() {
        this.warmPool = Caffeine.newBuilder()
                .maximumSize(MAX_POOL_SIZE)
                .expireAfterAccess(IDLE_TIMEOUT)
                .removalListener((RunnerCacheKey key, ModelRunner runner, RemovalCause cause) -> this
                        .onRunnerEvicted(key, runner, cause))
                .build(new CacheLoader<RunnerCacheKey, ModelRunner>() {
                    @Override
                    public ModelRunner load(RunnerCacheKey key) {
                        return createRunner(key);
                    }
                });

        this.lastAccess = new ConcurrentHashMap<>();

        // Start cleanup scheduler
        startCleanupScheduler();
    }

    /**
     * Get or create runner instance for tenant
     */
    public ModelRunner getRunner(
            ModelManifest manifest,
            String runnerName,
            TenantContext tenantContext) {
        RunnerCacheKey key = new RunnerCacheKey(
                tenantContext.getTenantId(),
                manifest.modelId(),
                runnerName);

        // Update last access time
        lastAccess.put(key, Instant.now());

        // Get from pool or create
        return warmPool.get(key);
    }

    /**
     * Prewarm runners for specific models
     */
    public void prewarm(
            ModelManifest manifest,
            List<String> runnerNames,
            TenantContext tenantContext) {
        runnerNames.forEach(runnerName -> {
            try {
                getRunner(manifest, runnerName, tenantContext);
            } catch (Exception e) {
                // Log but don't fail prewarming
                Log.warnf("Failed to prewarm runner %s: %s",
                        runnerName, e.getMessage());
            }
        });
    }

    /**
     * Create new runner instance (called by cache loader)
     */
    private ModelRunner createRunner(RunnerCacheKey key) {
        // Load manifest
        ModelManifest manifest = repository
                .findById(key.modelId(), key.tenantId())
                .orElseThrow(() -> new ModelNotFoundException(key.modelId()));

        // Find runner provider by name
        ModelRunnerProvider provider = runnerProviders.stream()
                .filter(r -> r.metadata().name().equals(key.runnerName()))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException(
                        "Unknown runner: " + key.runnerName()));

        // Get tenant-specific configuration
        Map<String, Object> config = tenantConfigService.getRunnerConfig(
                key.tenantId().value(),
                key.runnerName());

        // Initialize runner
        TenantContext ctx = TenantContext.of(key.tenantId());
        ModelRunner runner = provider.create(manifest, config, ctx);

        // Warmup if configured
        if (config.getOrDefault("warmup.enabled", false).equals(true)) {
            runner.warmup(Collections.emptyList());
        }

        Log.infof("Created runner %s for model %s (tenant %s)",
                key.runnerName(), key.modelId(), key.tenantId().value());

        return runner;
    }

    /**
     * Cleanup callback when runner is evicted
     */
    private void onRunnerEvicted(
            RunnerCacheKey key,
            ModelRunner runner,
            RemovalCause cause) {
        if (runner != null) {
            try {
                runner.close();
                Log.infof("Closed runner %s for model %s (cause: %s)",
                        key.runnerName(), key.modelId(), cause);
            } catch (Exception e) {
                Log.errorf(e, "Error closing runner %s", key.runnerName());
            }
        }
        lastAccess.remove(key);
    }

    /**
     * Background cleanup of idle runners
     */
    private void startCleanupScheduler() {
        this.scheduler = Executors.newSingleThreadScheduledExecutor();
        this.scheduler.scheduleAtFixedRate(
                this::cleanupIdleRunners,
                5, 5, TimeUnit.MINUTES);
    }

    @PreDestroy
    public void shutdown() {
        if (scheduler != null) {
            scheduler.shutdown();
            try {
                if (!scheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                    scheduler.shutdownNow();
                }
            } catch (InterruptedException e) {
                scheduler.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
        warmPool.invalidateAll();
    }

    private void cleanupIdleRunners() {
        Instant threshold = Instant.now().minus(IDLE_TIMEOUT);

        lastAccess.entrySet().removeIf(entry -> {
            if (entry.getValue().isBefore(threshold)) {
                warmPool.invalidate(entry.getKey());
                return true;
            }
            return false;
        });
    }

    /**
     * Cache key for runner pooling
     */
    private record RunnerCacheKey(
            TenantId tenantId,
            String modelId,
            String runnerName) {
    }
}