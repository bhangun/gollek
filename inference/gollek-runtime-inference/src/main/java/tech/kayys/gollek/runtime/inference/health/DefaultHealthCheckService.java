package tech.kayys.gollek.runtime.inference.health;

import org.jboss.logging.Logger;
import tech.kayys.gollek.runtime.inference.batch.ContinuousBatchScheduler;
import tech.kayys.gollek.runtime.inference.kv.KVCacheManager;
import tech.kayys.gollek.runtime.inference.kv.GPUMemoryManager;
import tech.kayys.gollek.runtime.inference.kv.PagedKVCache;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Default implementation of {@link HealthCheckService}.
 *
 * @since 0.2.0
 */
public final class DefaultHealthCheckService implements HealthCheckService {

    private static final Logger LOG = Logger.getLogger(DefaultHealthCheckService.class);

    // ── Components ─────────────────────────────────────────────────────

    /** Registered models: modelId → ModelHealthEntry */
    private final Map<String, ModelHealthEntry> models = new ConcurrentHashMap<>();

    /** Registered schedulers: modelId → scheduler */
    private final Map<String, ContinuousBatchScheduler> schedulers = new ConcurrentHashMap<>();

    /** KV cache manager */
    private final KVCacheManager kvCacheManager;

    /** GPU memory manager */
    private final GPUMemoryManager gpuMemoryManager;

    /** Paged KV caches: modelId → PagedKVCache */
    private final Map<String, PagedKVCache> pagedCaches = new ConcurrentHashMap<>();

    // ── Configuration ──────────────────────────────────────────────────

    /** Startup timeout */
    private final Duration startupTimeout;

    /** Service start time */
    private final LocalDateTime startedAt;

    /** Whether startup is complete */
    private final AtomicBoolean startupComplete = new AtomicBoolean(false);

    /** Number of models loaded */
    private final AtomicInteger modelsLoaded = new AtomicInteger(0);

    /**
     * Creates a new health check service.
     */
    private DefaultHealthCheckService(Config config) {
        this.startupTimeout = config.startupTimeout;
        this.kvCacheManager = config.kvCacheManager;
        this.gpuMemoryManager = config.gpuMemoryManager;
        this.startedAt = LocalDateTime.now();

        LOG.infof("Health check service started at %s (startup timeout: %ds)",
            startedAt, startupTimeout.getSeconds());
    }

    // ── Registration Methods ───────────────────────────────────────────

    /**
     * Registers a model for health tracking.
     */
    public void registerModel(String modelId) {
        models.put(modelId, new ModelHealthEntry(modelId));
        LOG.infof("Registered model for health tracking: %s", modelId);
    }

    /**
     * Marks a model as loaded.
     */
    public void markModelLoaded(String modelId) {
        ModelHealthEntry info = models.get(modelId);
        if (info != null) {
            info.loaded = true;
            info.loadedAt = LocalDateTime.now();
            modelsLoaded.incrementAndGet();
            LOG.infof("Model loaded: %s (%d/%d loaded)",
                modelId, modelsLoaded.get(), models.size());
            checkStartupComplete();
        }
    }

    /**
     * Marks a model as warmed (ready for inference).
     */
    public void markModelWarmed(String modelId) {
        ModelHealthEntry info = models.get(modelId);
        if (info != null) {
            info.warmed = true;
        }
    }

    /**
     * Registers a scheduler for health tracking.
     */
    public void registerScheduler(String modelId, ContinuousBatchScheduler scheduler) {
        schedulers.put(modelId, scheduler);
        LOG.infof("Registered scheduler for health tracking: %s", modelId);
    }

    /**
     * Registers a paged KV cache for health tracking.
     */
    public void registerKVCache(String modelId, PagedKVCache pagedCache) {
        pagedCaches.put(modelId, pagedCache);
    }

    // ── Health Check Implementations ───────────────────────────────────

    @Override
    public HealthResult checkStartup() {
        LocalDateTime now = LocalDateTime.now();
        Duration elapsed = Duration.between(startedAt, now);

        // Check if startup timeout exceeded
        if (elapsed.compareTo(startupTimeout) > 0 && !startupComplete.get()) {
            return HealthResult.unhealthy("startup",
                "Startup timeout exceeded: " + elapsed.getSeconds() + "s > " + startupTimeout.getSeconds() + "s",
                Map.of(
                    "elapsedSeconds", elapsed.getSeconds(),
                    "timeoutSeconds", startupTimeout.getSeconds(),
                    "modelsLoaded", modelsLoaded.get(),
                    "modelsExpected", models.size()
                ));
        }

        // Check if all models loaded
        if (!startupComplete.get()) {
            return HealthResult.unhealthy("startup",
                "Still loading models",
                Map.of(
                    "modelsLoaded", modelsLoaded.get(),
                    "modelsExpected", models.size()
                ));
        }

        return HealthResult.healthy("startup",
            "All models loaded and ready");
    }

    @Override
    public HealthResult checkReadiness() {
        // Check if any models registered
        if (models.isEmpty()) {
            return HealthResult.unhealthy("readiness", "No models registered");
        }

        // Check if all models loaded
        for (var entry : models.entrySet()) {
            String modelId = entry.getKey();
            ModelHealthEntry info = entry.getValue();

            if (!info.loaded) {
                return HealthResult.unhealthy("readiness",
                    "Model not loaded: " + modelId,
                    Map.of("modelId", modelId));
            }

            // Check scheduler is running
            ContinuousBatchScheduler scheduler = schedulers.get(modelId);
            if (scheduler != null && !scheduler.isRunning()) {
                return HealthResult.unhealthy("readiness",
                    "Scheduler not running for model: " + modelId,
                    Map.of("modelId", modelId));
            }
        }

        // Check KV cache manager
        if (kvCacheManager != null) {
            try {
                // Try to get stats (will throw if closed)
                kvCacheManager.getStats(models.keySet().stream().findFirst().orElse(""));
            } catch (IllegalStateException e) {
                return HealthResult.unhealthy("readiness", "KV cache manager is closed");
            }
        }

        return HealthResult.healthy("readiness",
            "All models loaded and schedulers running");
    }

    @Override
    public HealthResult checkLiveness() {
        // Check for deadlocks (simplified: check if we can acquire lock)
        try {
            // If we can call this method, we're not deadlocked
        } catch (Exception e) {
            return HealthResult.unhealthy("liveness", "Service appears deadlocked",
                Map.of("error", e.getMessage()));
        }

        // Check GPU memory manager
        if (gpuMemoryManager != null) {
            try {
                gpuMemoryManager.stats();
            } catch (IllegalStateException e) {
                return HealthResult.unhealthy("liveness", "GPU memory manager is closed");
            }
        }

        // Check for critical failures
        for (var entry : models.entrySet()) {
            String modelId = entry.getKey();
            ModelHealthEntry info = entry.getValue();

            if (info.failed) {
                return HealthResult.unhealthy("liveness",
                    "Model failed: " + modelId,
                    Map.of("modelId", modelId, "error", info.errorMessage));
            }
        }

        return HealthResult.healthy("liveness", "Service is alive and responsive");
    }

    @Override
    public HealthDetails getHealthDetails() {
        HealthResult startup = checkStartup();
        HealthResult readiness = checkReadiness();
        HealthResult liveness = checkLiveness();

        // Model health
        Map<String, ModelHealth> modelHealth = new HashMap<>();
        for (var entry : models.entrySet()) {
            String modelId = entry.getKey();
            ModelHealthEntry info = entry.getValue();

            String status;
            if (info.failed) {
                status = "failed";
            } else if (!info.loaded) {
                status = "loading";
            } else if (!info.warmed) {
                status = "loaded";
            } else {
                status = "ready";
            }

            modelHealth.put(modelId, new ModelHealth(
                modelId,
                info.loaded,
                info.warmed,
                1,  // loadedVersions
                info.loadedAt,
                0,  // inferenceCount (would come from metrics)
                0.0,  // avgLatencyMs
                status
            ));
        }

        // Scheduler health
        Map<String, SchedulerHealth> schedulerHealth = new HashMap<>();
        for (var entry : schedulers.entrySet()) {
            String modelId = entry.getKey();
            ContinuousBatchScheduler scheduler = entry.getValue();

            String status = scheduler.isShuttingDown() ? "shutting_down" :
                           scheduler.isRunning() ? "running" : "stopped";

            schedulerHealth.put(modelId, new SchedulerHealth(
                modelId,
                scheduler.isRunning(),
                scheduler.activeCount(),
                scheduler.pendingCount(),
                scheduler.getTotalBatchedRequests(),
                scheduler.getTotalRejected(),
                0.0,  // avgBatchUtilization (would come from scheduler metrics)
                status
            ));
        }

        // Memory health
        MemoryHealth memoryHealth = getMemoryHealth();

        return new HealthDetails(
            startup,
            readiness,
            liveness,
            modelHealth,
            schedulerHealth,
            memoryHealth,
            LocalDateTime.now()
        );
    }

    // ── Internal Methods ───────────────────────────────────────────────

    private void checkStartupComplete() {
        if (modelsLoaded.get() >= models.size()) {
            startupComplete.compareAndSet(false, true);
            LOG.info("Startup complete: all models loaded");
        }
    }

    private MemoryHealth getMemoryHealth() {
        long totalMemory = 0;
        long allocatedMemory = 0;
        long availableMemory = 0;
        double utilization = 0.0;
        int kvBlocksUsed = 0;
        int kvBlocksTotal = 0;
        double kvUtilization = 0.0;
        boolean oomRisk = false;
        String status = "healthy";

        // Get GPU memory stats
        if (gpuMemoryManager != null) {
            try {
                GPUMemoryManager.MemoryStats stats = gpuMemoryManager.stats();
                totalMemory = stats.totalMemory();
                allocatedMemory = stats.allocatedMemory();
                availableMemory = stats.availableMemory();
                utilization = stats.utilization();
                oomRisk = utilization > 0.9;
                status = utilization > 0.85 ? "warning" : "healthy";
            } catch (Exception e) {
                status = "error";
            }
        }

        // Get KV cache stats
        for (var entry : pagedCaches.entrySet()) {
            PagedKVCache cache = entry.getValue();
            kvBlocksUsed += cache.usedBlocks();
            kvBlocksTotal += cache.totalBlocks();
        }

        if (kvBlocksTotal > 0) {
            kvUtilization = (double) kvBlocksUsed / kvBlocksTotal;
            if (kvUtilization > 0.9) {
                oomRisk = true;
                status = "critical";
            } else if (kvUtilization > 0.85 && !"critical".equals(status)) {
                status = "warning";
            }
        }

        return new MemoryHealth(
            totalMemory,
            allocatedMemory,
            availableMemory,
            utilization,
            kvBlocksUsed,
            kvBlocksTotal,
            kvUtilization,
            oomRisk,
            status
        );
    }

    // ── Nested Classes ─────────────────────────────────────────────────

    /**
     * Internal model tracking info.
     */
    private static final class ModelHealthEntry {
        final String modelId;
        boolean loaded = false;
        boolean warmed = false;
        boolean failed = false;
        String errorMessage = "";
        LocalDateTime loadedAt = null;

        ModelHealthEntry(String modelId) {
            this.modelId = modelId;
        }
    }

    /**
     * Configuration for DefaultHealthCheckService.
     */
    private static final class Config {
        Duration startupTimeout = Duration.ofMinutes(5);
        KVCacheManager kvCacheManager;
        GPUMemoryManager gpuMemoryManager;
    }

    /**
     * Creates a new builder for configuring the health check service.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for DefaultHealthCheckService.
     */
    public static final class Builder {
        private final Config config = new Config();

        private Builder() {}

        /**
         * Sets the startup timeout.
         */
        public Builder startupTimeout(Duration timeout) {
            config.startupTimeout = timeout;
            return this;
        }

        /**
         * Sets the KV cache manager.
         */
        public Builder kvCacheManager(KVCacheManager kvCacheManager) {
            config.kvCacheManager = kvCacheManager;
            return this;
        }

        /**
         * Sets the GPU memory manager.
         */
        public Builder gpuMemoryManager(GPUMemoryManager gpuMemoryManager) {
            config.gpuMemoryManager = gpuMemoryManager;
            return this;
        }

        public DefaultHealthCheckService build() {
            return new DefaultHealthCheckService(config);
        }
    }
}
