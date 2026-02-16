package tech.kayys.golek.inference.gguf;

import jakarta.annotation.PreDestroy;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;

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
public class GGUFSessionManager {

    private static final Logger log = Logger.getLogger(GGUFSessionManager.class);

    private final LlamaCppBinding binding;
    private final GGUFChatTemplateService templateService;

    @Inject
    public GGUFSessionManager(LlamaCppBinding binding, GGUFChatTemplateService templateService) {
        this.binding = binding;
        this.templateService = templateService;
    }

    private final Map<String, SessionPool> pools = new ConcurrentHashMap<>();
    private final AtomicInteger totalActiveSessions = new AtomicInteger(0);
    private volatile boolean initialized = false;
    private volatile boolean shutdown = false;

    /**
     * Session context wrapper
     */
    public record SessionContext(
            String sessionId,
            LlamaCppRunner runner,
            Instant createdAt,
            Instant lastUsedAt) {
        public SessionContext touch() {
            return new SessionContext(sessionId, runner, createdAt, Instant.now());
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
        private final String modelId;
        private final GGUFProviderConfig config;
        private final Map<String, SessionContext> sessions = new ConcurrentHashMap<>();
        private final Semaphore permits;

        SessionPool(String poolKey, String modelId, GGUFProviderConfig config) {
            this.poolKey = poolKey;
            this.modelId = modelId;
            this.config = config;
            this.permits = new Semaphore(config.sessionPoolMaxSize(), true);
        }

        SessionContext acquire() throws InterruptedException {
            permits.acquire();

            try {
                // Try to find an idle session
                SessionContext session = findIdleSession();
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

        private SessionContext findIdleSession() {
            return sessions.values().stream()
                    .filter(s -> !s.isIdle(config.sessionPoolIdleTimeout()))
                    .findFirst()
                    .orElse(null);
        }

        private SessionContext createSession() {
            String sessionId = java.util.UUID.randomUUID().toString();

            // Extract tenant id from poolKey
            String requestId = poolKey.split(":")[0];

            // Create artifact location
            tech.kayys.golek.spi.model.ArtifactLocation location = new tech.kayys.golek.spi.model.ArtifactLocation(
                    resolveModelPath(modelId, config),
                    null,
                    null,
                    null);

            // Create model manifest
            tech.kayys.golek.spi.model.ModelManifest manifest = tech.kayys.golek.spi.model.ModelManifest.builder()
                    .modelId(modelId)
                    .name(modelId)
                    .version("unknown")
                    .path(location.uri())
                    .apiKey(tech.kayys.golek.spi.auth.ApiKeyConstants.COMMUNITY_API_KEY)
                    .requestId(requestId)
                    .artifacts(Map.of(tech.kayys.golek.spi.model.ModelFormat.GGUF, location))
                    .supportedDevices(java.util.Collections.emptyList())
                    .resourceRequirements(null)
                    .metadata(java.util.Collections.emptyMap())
                    .createdAt(Instant.now())
                    .updatedAt(Instant.now())
                    .build();

            // Create runner configuration
            Map<String, Object> runnerConfig = Map.of(
                    "nGpuLayers", config.gpuEnabled() ? config.gpuLayers() : 0,
                    "nThreads", config.threads(),
                    "nCtx", config.maxContextTokens(),
                    "nBatch", config.batchSize(),
                    "useMmap", config.mmapEnabled(),
                    "useMlock", config.mlockEnabled());

            // Create and initialize runner
            LlamaCppRunner runner = new LlamaCppRunner(binding, config, templateService);

            try {
                runner.initialize(manifest, runnerConfig);

                // Warmup runner if configured
                if (config.prewarmEnabled()) {
                    runner.warmup(runner.createDefaultWarmupRequests());
                }

            } catch (Exception e) {
                runner.close();
                throw new RuntimeException("Failed to initialize runner: " + e.getMessage(), e);
            }

            return new SessionContext(
                    sessionId,
                    runner,
                    Instant.now(),
                    Instant.now());
        }

        void cleanup() {
            Duration timeout = config.sessionPoolIdleTimeout();
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
    public SessionContext getSession(String requestId, String modelId, GGUFProviderConfig config) {
        ensureInitialized();

        String poolKey = buildPoolKey(requestId, modelId);

        // Get or create pool
        SessionPool pool = pools.computeIfAbsent(
                poolKey,
                k -> new SessionPool(k, modelId, config));

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
        String poolKey = buildPoolKey(requestId, modelId);
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
        pools.values().forEach(SessionPool::shutdown);
        pools.clear();

        log.infof("GGUF Session Manager shutdown complete (cleaned up %d sessions)",
                totalActiveSessions.get());
    }

    private String buildPoolKey(String requestId, String modelId) {
        return requestId + ":" + modelId;
    }

    private String resolveModelPath(String modelId, GGUFProviderConfig config) {
        if (modelId == null)
            return null;
        if (modelId.startsWith("/"))
            return modelId;

        String normalizedId = modelId.replace("/", "_");
        String basePath = config.modelBasePath();
        java.nio.file.Path modelDir = java.nio.file.Paths.get(basePath);

        // Try variations
        String[] variations = {
                normalizedId,
                normalizedId + "-GGUF",
                modelId,
                modelId + "-GGUF"
        };

        for (String var : variations) {
            java.nio.file.Path p = modelDir.resolve(var);
            if (java.nio.file.Files.exists(p)) {
                return p.toString();
            }
        }

        return modelDir.resolve(normalizedId).toString();
    }

    private void startCleanupTask() {
        // TODO: Implement periodic cleanup task using Quarkus Scheduler
        // For now, cleanup is manual via cleanupIdleSessions()
    }

    /**
     * Manual cleanup trigger (can be called by scheduler)
     */
    public void cleanupIdleSessions() {
        if (shutdown) {
            return;
        }

        log.debug("Running idle session cleanup");
        pools.values().forEach(SessionPool::cleanup);

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
}
