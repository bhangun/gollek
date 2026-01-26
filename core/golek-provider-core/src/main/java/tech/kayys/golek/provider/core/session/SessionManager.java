package tech.kayys.golek.provider.core.session;

import org.jboss.logging.Logger;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.*;

/**
 * Manages session pools for all models
 */
public class SessionManager {

    private static final Logger LOG = Logger.getLogger(SessionManager.class);

    private final SessionConfig config;
    private final Map<String, SessionPool> pools;
    private final ScheduledExecutorService cleanupExecutor;
    private volatile boolean shutdown;

    public SessionManager(SessionConfig config) {
        this.config = config;
        this.pools = new ConcurrentHashMap<>();
        this.cleanupExecutor = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "session-cleanup");
            t.setDaemon(true);
            return t;
        });
        this.shutdown = false;

        // Schedule periodic cleanup
        cleanupExecutor.scheduleAtFixedRate(
                this::cleanupAllPools,
                1, 5, TimeUnit.MINUTES);
    }

    /**
     * Acquire session for model and tenant
     */
    public Optional<Session> acquireSession(String modelId, String tenantId) {
        return acquireSession(modelId, tenantId, 5000);
    }

    /**
     * Acquire session with timeout
     */
    public Optional<Session> acquireSession(
            String modelId,
            String tenantId,
            long timeoutMs) {
        if (shutdown) {
            return Optional.empty();
        }

        String poolKey = getPoolKey(modelId, tenantId);
        SessionPool pool = pools.computeIfAbsent(poolKey,
                k -> createPool(modelId, tenantId));

        return pool.acquire(timeoutMs);
    }

    /**
     * Release session back to pool
     */
    public void releaseSession(Session session) {
        if (session == null || shutdown) {
            return;
        }

        String poolKey = getPoolKey(session.getModelId(), session.getTenantId());
        SessionPool pool = pools.get(poolKey);

        if (pool != null) {
            pool.release(session);
        } else {
            LOG.warnf("No pool found for session %s, closing", session.getSessionId());
            session.close();
        }
    }

    /**
     * Get active session count
     */
    public int activeSessionCount() {
        return pools.values().stream()
                .mapToInt(pool -> pool.getStats().activeSessions())
                .sum();
    }

    /**
     * Get total session count
     */
    public int totalSessionCount() {
        return pools.values().stream()
                .mapToInt(pool -> pool.getStats().totalSessions())
                .sum();
    }

    /**
     * Get pool statistics for model
     */
    public Optional<SessionPool.PoolStats> getPoolStats(String modelId, String tenantId) {
        String poolKey = getPoolKey(modelId, tenantId);
        SessionPool pool = pools.get(poolKey);
        return pool != null ? Optional.of(pool.getStats()) : Optional.empty();
    }

    /**
     * Cleanup all pools
     */
    public void cleanupAllPools() {
        int totalCleaned = 0;
        for (SessionPool pool : pools.values()) {
            totalCleaned += pool.cleanup();
        }

        if (totalCleaned > 0) {
            LOG.infof("Cleaned %d idle/expired sessions across all pools", totalCleaned);
        }
    }

    /**
     * Shutdown manager and all pools
     */
    public void shutdown() {
        if (shutdown) {
            return;
        }

        shutdown = true;

        LOG.info("Shutting down session manager");

        cleanupExecutor.shutdownNow();

        for (SessionPool pool : pools.values()) {
            pool.shutdown();
        }

        pools.clear();

        LOG.info("Session manager shutdown complete");
    }

    private SessionPool createPool(String modelId, String tenantId) {
        LOG.infof("Creating session pool for model %s, tenant %s", modelId, tenantId);
        return new SessionPool(modelId, tenantId, config);
    }

    private String getPoolKey(String modelId, String tenantId) {
        return modelId + ":" + tenantId;
    }
}