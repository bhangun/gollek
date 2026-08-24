package tech.kayys.gollek.inference.llamacpp;



import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import jakarta.annotation.PreDestroy;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.observability.AdapterMetricSchema;
import tech.kayys.gollek.spi.observability.AdapterMetricsRecorder;


import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Manages GGUF model sessions with pooling and tenant isolation.
 * 
 * Features:
 * - Per-tenant/model session pooling
 * - Configurable pool sizes (min/max)
 * - Idle timeout and cleanup
 * - Resource limits enforcement
 * - Thread-safe concurrent access
 * 
 * Thread-safety: All methods are thread-safe.
 */
@ApplicationScoped
public class LlamaCppSessionManager {

    private static final Logger log = Logger.getLogger(LlamaCppSessionManager.class);
    private static final Duration DEFAULT_IDLE_TIMEOUT = Duration.ofMinutes(5);
    private static final long MIN_CLEANUP_INTERVAL_SECONDS = 10L;

    private final LlamaCppBinding binding;
    private final GGUFChatTemplateService templateService;
    @Inject
    AdapterMetricsRecorder adapterMetricsRecorder;
    @Inject
    MeterRegistry meterRegistry;

    private final tech.kayys.gollek.model.repo.local.ManifestStore manifestStore;

    @Inject
    public LlamaCppSessionManager(LlamaCppBinding binding, GGUFChatTemplateService templateService,
            tech.kayys.gollek.model.repo.local.ManifestStore manifestStore) {
        this.binding = binding;
        this.templateService = templateService;
        this.manifestStore = manifestStore;
    }

    private final Map<String, SessionPool> pools = new ConcurrentHashMap<>();
    private final AtomicInteger totalActiveSessions = new AtomicInteger(0);
    private volatile ScheduledExecutorService cleanupExecutor;
    private volatile boolean initialized = false;
    private volatile boolean shutdown = false;

    /**
     * Session context wrapper
     */
    public record SessionContext(
            String sessionId,
            LlamaCppRunner runner,
            AdapterSpec adapterSpec,
            Instant createdAt,
            Instant lastUsedAt) {
        public SessionContext touch() {
            return new SessionContext(sessionId, runner, adapterSpec, createdAt, Instant.now());
        }

        public boolean isIdle(Duration timeout) {
            return Duration.between(lastUsedAt, Instant.now()).compareTo(timeout) > 0;
        }
    }

    /**
     * Session pool for a specific tenant/model combination
     */
    private class SessionPool {
        private final String poolKey;
        private final String requestId;
        private final String modelId;
        private final AdapterSpec adapterSpec;
        private final LlamaCppProviderConfig config;
        private final Map<String, SessionContext> sessions = new ConcurrentHashMap<>();
        private final Semaphore permits;

        SessionPool(String poolKey, String requestId, String modelId, AdapterSpec adapterSpec,
                LlamaCppProviderConfig config) {
            this.poolKey = poolKey;
            this.requestId = requestId;
            this.modelId = modelId;
            this.adapterSpec = adapterSpec;
            this.config = config;
            this.permits = new Semaphore(config.sessionPoolMaxSize(), true);
        }

        SessionContext acquire() throws InterruptedException {
            permits.acquire();

            try {
                // Try to find an idle session
                SessionContext session = findIdleSession(resolveAdaptiveIdleTimeout(config));
                if (session != null) {
                    log.debugf("Reusing session %s for pool %s", session.sessionId(), poolKey);
                    return session.touch();
                }

                // Create new session
                session = createSession();
                sessions.put(session.sessionId(), session);
                totalActiveSessions.incrementAndGet();

                log.debugf(" new session %s for pool %s (total active: %d)",
                        session.sessionId(), poolKey, totalActiveSessions.get());

                return session;

            } catch (Exception e) {
                permits.release();
                throw e;
            }
        }

        void release(SessionContext session) {
            // Update last used timestamp
            sessions.put(session.sessionId(), session.touch());
            permits.release();
        }

        private SessionContext findIdleSession(Duration timeout) {
            return sessions.values().stream()
                    .filter(s -> !s.isIdle(timeout))
                    .findFirst()
                    .orElse(null);
        }

        private SessionContext createSession() {
            String sessionId = java.util.UUID.randomUUID().toString();

            // Create artifact location
            tech.kayys.alkhawarizm.spi.model.ArtifactLocation location = new tech.kayys.alkhawarizm.spi.model.ArtifactLocation(
                    resolveModelPath(modelId, config),
                    null,
                    null,
                    null);

            // Create model manifest
            tech.kayys.alkhawarizm.spi.model.ModelManifest manifest = tech.kayys.alkhawarizm.spi.model.ModelManifest.builder()
                    .modelId(modelId)
                    .name(modelId)
                    .version("unknown")
                    .path(location.uri())
                    .apiKey(tech.kayys.gollek.spi.auth.ApiKeyConstants.COMMUNITY_API_KEY)
                    .requestId(requestId)
                    .artifacts(Map.of(tech.kayys.alkhawarizm.core.model.ModelFormat.GGUF, location))
                    .supportedDevices(java.util.Collections.emptyList())
                    .resourceRequirements(null)
                    .metadata(java.util.Collections.emptyMap())
                    .createdAt(Instant.now())
                    .updatedAt(Instant.now())
                    .build();

            // Create runner configuration
            int effectiveGpuLayers = LlamaCppDeviceSupport.resolveGpuLayers(config);
            Map<String, Object> runnerConfig = Map.of(
                    "nGpuLayers", effectiveGpuLayers,
                    "nThreads", config.threads(),
                    "nCtx", config.maxContextTokens(),
                    "nBatch", config.batchSize(),
                    "useMmap", config.mmapEnabled(),
                    "useMlock", config.mlockEnabled());
            if (adapterSpec != null) {
                runnerConfig = new java.util.HashMap<>(runnerConfig);
                runnerConfig.put("adapter.type", adapterSpec.type());
                runnerConfig.put("adapter.id", adapterSpec.adapterId());
                runnerConfig.put("adapter.path", adapterSpec.adapterPath());
                runnerConfig.put("adapter.scale", adapterSpec.scale());
            }

            // Create and initialize runner
            LlamaCppRunner runner = new LlamaCppRunner(binding, config, templateService);

            try {
                Instant initStart = Instant.now();
                runner.initialize(manifest, runnerConfig);
                if (config.metricsEnabled() && adapterSpec != null && adapterMetricsRecorder != null) {
                    var schema = AdapterMetricSchema.builder()
                            .adapterId(adapterSpec.adapterId())
                            .operation("adapter_apply")
                            .build();
                    adapterMetricsRecorder.recordSuccess(schema, Duration.between(initStart, Instant.now()).toMillis());
                }

                // Warmup runner if configured
                if (config.prewarmEnabled()) {
                    runner.warmup(runner.createDefaultWarmupRequests());
                }
                if (meterRegistry != null) {
                    runner.registerMetrics(meterRegistry, requestId, modelId);
                }

            } catch (Exception e) {
                runner.close();
                throw new RuntimeException("Failed to initialize runner: " + e.getMessage(), e);
            }

            return new SessionContext(
                    sessionId,
                    runner,
                    adapterSpec,
                    Instant.now(),
                    Instant.now());
        }

        void cleanup(Duration timeout) {
            int minSize = config.sessionPoolMinSize();

            var it = sessions.entrySet().iterator();
            int retained = 0;

            while (it.hasNext()) {
                var entry = it.next();
                SessionContext session = entry.getValue();

                // Keep at least min size sessions
                if (retained < minSize) {
                    retained++;
                    continue;
                }

                // Remove idle sessions
                if (session.isIdle(timeout)) {
                    log.debugf("Cleaning up idle session %s from pool %s",
                            session.sessionId(), poolKey);

                    session.runner().close();
                    it.remove();
                    totalActiveSessions.decrementAndGet();
                }
            }
        }

        void shutdown() {
            log.debugf("Shutting down session pool %s with %d sessions",
                    poolKey, sessions.size());

            sessions.values().forEach(session -> {
                try {
                    session.runner().close();
                    totalActiveSessions.decrementAndGet();
                } catch (Exception e) {
                    log.warnf(e, "Error closing session %s", session.sessionId());
                }
            });

            sessions.clear();
        }

        int size() {
            return sessions.size();
        }
    }

    /**
     * Initialize session manager
     */
    public void initialize() {
        if (initialized) {
            return;
        }

        log.info("Initializing GGUF Session Manager");

        // Start cleanup task
        startCleanupTask();

        initialized = true;
        log.info("GGUF Session Manager initialized");
    }

    /**
     * Get or create a session for the given tenant/model
     * 
     * @param requestId Tenant identifier
     * @param modelId   Model identifier
     * @param config    Provider configuration
     * @return Session context
     */
    public SessionContext getSession(String requestId, String modelId, LlamaCppProviderConfig config) {
        return getSession(requestId, modelId, config, null);
    }

    public SessionContext getSession(String requestId, String modelId, LlamaCppProviderConfig config,
            AdapterSpec adapterSpec) {
        ensureInitialized();

        String poolKey = buildPoolKey(requestId, modelId, adapterSpec);

        // Get or create pool
        SessionPool pool = pools.computeIfAbsent(
                poolKey,
                k -> new SessionPool(k, requestId, modelId, adapterSpec, config));

        try {
            return pool.acquire();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted while acquiring session", e);
        }
    }

    /**
     * Release a session back to the pool
     */
    public void releaseSession(String requestId, String modelId, SessionContext session) {
        releaseSession(requestId, modelId, session, session != null ? session.adapterSpec() : null);
    }

    public void releaseSession(String requestId, String modelId, SessionContext session, AdapterSpec adapterSpec) {
        String poolKey = buildPoolKey(requestId, modelId, adapterSpec);
        SessionPool pool = pools.get(poolKey);

        if (pool != null) {
            pool.release(session);
        } else {
            log.warnf("No pool found for %s, closing session", poolKey);
            session.runner().close();
        }
    }

    /**
     * Get total number of active sessions across all pools
     */
    public int getActiveSessionCount() {
        return totalActiveSessions.get();
    }

    /**
     * Check if session manager is healthy
     */
    public boolean isHealthy() {
        return initialized && !shutdown;
    }

    /**
     * Graceful shutdown
     */
    @PreDestroy
    public void shutdown() {
        if (shutdown) {
            return;
        }

        log.info("Shutting down GGUF Session Manager");
        shutdown = true;

        // Shutdown all pools
        if (cleanupExecutor != null) {
            cleanupExecutor.shutdownNow();
        }
        pools.values().forEach(SessionPool::shutdown);
        pools.clear();

        log.infof("GGUF Session Manager shutdown complete (cleaned up %d sessions)",
                totalActiveSessions.get());
    }

    private String buildPoolKey(String requestId, String modelId) {
        return buildPoolKey(requestId, modelId, null);
    }

    private String buildPoolKey(String requestId, String modelId, AdapterSpec adapterSpec) {
        String adapterKey = adapterSpec == null ? "no-adapter" : adapterSpec.cacheKey();
        return requestId + ":" + modelId + ":" + adapterKey;
    }

    private String resolveModelPath(String modelId, LlamaCppProviderConfig config) {
        if (modelId == null) return null;
        if (modelId.startsWith("/")) return modelId;

        // 1. Try manifest resolution
        var manifest = manifestStore.findByModelId(modelId, "gguf");
        if (manifest.isPresent()) {
            return manifest.get().getBlobPath();
        }

        // 2. Fallback to name-based lookup
        String normalizedId = modelId.replace("/", "__").replace(".", "_");
        manifest = manifestStore.findByName(normalizedId);
        if (manifest.isPresent()) {
            return manifest.get().getBlobPath();
        }

        log.warnf("GGUF session model not found in manifest: %s. Falling back to default path calculation.", modelId);
        return tech.kayys.gollek.model.repo.local.ManifestStore.getModelsRoot().resolve("blobs").resolve(normalizedId).toAbsolutePath().toString();
    }

    private void startCleanupTask() {
        if (cleanupExecutor != null) {
            return;
        }

        long idleSeconds = pools.values().stream()
                .findFirst()
                .map(pool -> Math.max(1L, pool.config.sessionPoolIdleTimeout().getSeconds()))
                .orElse(Math.max(1L, DEFAULT_IDLE_TIMEOUT.getSeconds()));
        long intervalSeconds = Math.max(MIN_CLEANUP_INTERVAL_SECONDS, idleSeconds / 2L);
        cleanupExecutor = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "gollek-gguf-session-cleanup");
            t.setDaemon(true);
            return t;
        });
        cleanupExecutor.scheduleAtFixedRate(this::cleanupIdleSessions, intervalSeconds, intervalSeconds,
                TimeUnit.SECONDS);
        log.infof("Started GGUF session cleanup task (interval=%ds)", intervalSeconds);
    }

    /**
     * Manual cleanup trigger (can be called by scheduler)
     */
    public void cleanupIdleSessions() {
        if (shutdown) {
            return;
        }

        Duration adaptiveTimeout = resolveAdaptiveIdleTimeout(null);
        log.debug("Running idle session cleanup");
        pools.values().forEach(pool -> pool.cleanup(adaptiveTimeout));

        // Remove empty pools
        pools.entrySet().removeIf(entry -> {
            if (entry.getValue().size() == 0) {
                log.debugf("Removing empty pool %s", entry.getKey());
                return true;
            }
            return false;
        });
    }

    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("Session manager not initialized");
        }
        if (shutdown) {
            throw new IllegalStateException("Session manager has been shutdown");
        }
    }

    private Duration resolveAdaptiveIdleTimeout(LlamaCppProviderConfig poolConfig) {
        LlamaCppProviderConfig effective = poolConfig;
        if (effective == null) {
            effective = pools.values().stream()
                    .findFirst()
                    .map(p -> p.config)
                    .orElse(null);
        }
        if (effective == null) {
            return DEFAULT_IDLE_TIMEOUT;
        }

        return effective.sessionPoolIdleTimeout();
    }
}
