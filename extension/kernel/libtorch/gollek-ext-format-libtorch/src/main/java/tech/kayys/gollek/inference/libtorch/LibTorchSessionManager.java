package tech.kayys.gollek.inference.libtorch;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.core.Device;

import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Manages a pool of {@link TorchScriptRunner} sessions per tenant and model.
 * <p>
 * Supports idle session reuse: released sessions are returned to an idle deque
 * rather than being closed immediately. A background evictor periodically
 * removes sessions that have been idle beyond the configured timeout.
 * <p>
 * Thread-safe. Pools are keyed by "{tenantId}:{modelId}".
 */
@ApplicationScoped
public class LibTorchSessionManager {

    private static final Logger log = Logger.getLogger(LibTorchSessionManager.class);

    private final Map<String, SessionPool> pools = new ConcurrentHashMap<>();
    private ScheduledExecutorService evictor;

    @Inject
    LibTorchProviderConfig config;

    /**
     * Start the idle session evictor. Called after CDI init.
     */
    public void startEvictor() {
        int idleTimeout = config.session().idleTimeoutSeconds();
        if (idleTimeout > 0) {
            evictor = Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "gollek-session-evictor");
                t.setDaemon(true);
                return t;
            });
            long interval = Math.max(idleTimeout / 2, 10);
            evictor.scheduleAtFixedRate(this::evictIdleSessions, interval, interval, TimeUnit.SECONDS);
            log.infof("Session evictor started (interval=%ds, idleTimeout=%ds)", interval, idleTimeout);
        }
    }

    /**
     * Get or create a session for the given tenant/model.
     *
     * @param tenantId tenant identifier
     * @param modelId  model identifier
     * @param config   provider configuration
     * @return session context
     */
    public SessionContext getSession(String tenantId, String modelId, LibTorchProviderConfig config) {
        Path modelPath = resolveModelPath(modelId, config);
        String poolKey = tenantId + ":" + modelId;
        SessionPool pool = pools.computeIfAbsent(poolKey,
                k -> new SessionPool(modelPath, getDevice()));

        return pool.acquire();
    }

    /**
     * @deprecated Use {@link #getSession(String, String, LibTorchProviderConfig)}
     */
    @Deprecated
    public SessionContext acquire(String tenantId, String modelId, Path modelPath) {
        String poolKey = tenantId + ":" + modelId;
        SessionPool pool = pools.computeIfAbsent(poolKey,
                k -> new SessionPool(modelPath, getDevice()));
        return pool.acquire();
    }

    /**
     * Release a session back to the pool.
     */
    public void releaseSession(String tenantId, String modelId, SessionContext session) {
        String poolKey = tenantId + ":" + modelId;
        SessionPool pool = pools.get(poolKey);
        if (pool != null) {
            pool.release(session);
        } else {
            // Pool was removed (shutdown), close the runner
            closeRunner(session);
        }
    }

    /**
     * @deprecated Use {@link #releaseSession(String, String, SessionContext)}
     */
    @Deprecated
    public void release(String tenantId, String modelId, SessionContext session) {
        releaseSession(tenantId, modelId, session);
    }

    public Path resolveModelPath(String modelId, LibTorchProviderConfig config) {
        String basePath = config.model().basePath();
        String extensions = config.model().extensions();

        for (String ext : extensions.split(",")) {
            Path path = java.nio.file.Paths.get(basePath, modelId + ext.trim());
            if (java.nio.file.Files.exists(path)) {
                return path;
            }
        }

        // Fallback for absolute paths or already-extensioned IDs
        Path directPath = java.nio.file.Paths.get(modelId);
        if (java.nio.file.Files.exists(directPath)) {
            return directPath;
        }

        throw new RuntimeException("Model not found: " + modelId + " in " + basePath);
    }

    /**
     * Get the device to use based on configuration.
     */
    private Device getDevice() {
        if (config.gpu().enabled()) {
            return Device.cuda(config.gpu().deviceIndex());
        }
        return Device.CPU;
    }

    /**
     * Evict sessions that have been idle beyond the configured timeout.
     */
    private void evictIdleSessions() {
        long idleThreshold = System.currentTimeMillis()
                - (config.session().idleTimeoutSeconds() * 1000L);

        for (var entry : pools.entrySet()) {
            int evicted = entry.getValue().evictIdle(idleThreshold);
            if (evicted > 0) {
                log.debugf("Evicted %d idle sessions from pool %s", evicted, entry.getKey());
            }
        }
    }

    /**
     * Shutdown all session pools and release resources.
     */
    public void shutdown() {
        log.info("Shutting down LibTorch session pools");
        if (evictor != null) {
            evictor.shutdownNow();
        }
        pools.values().forEach(SessionPool::shutdown);
        pools.clear();
    }

    /**
     * Get the number of active sessions across all pools.
     */
    public int activeSessionCount() {
        return pools.values().stream()
                .mapToInt(SessionPool::activeCount)
                .sum();
    }

    /**
     * Get the number of idle sessions across all pools.
     */
    public int idleSessionCount() {
        return pools.values().stream()
                .mapToInt(SessionPool::idleCount)
                .sum();
    }

    /**
     * Get the total number of sessions (active + idle) across all pools.
     */
    public int totalSessionCount() {
        return activeSessionCount() + idleSessionCount();
    }

    /**
     * Get the total number of sessions ever created across all pools.
     */
    public int totalCreatedCount() {
        return pools.values().stream()
                .mapToInt(SessionPool::totalCreated)
                .sum();
    }

    private static void closeRunner(SessionContext ctx) {
        try {
            if (!ctx.runner().isClosed()) {
                ctx.runner().close();
            }
        } catch (Exception e) {
            log.warnf(e, "Error closing session runner");
        }
    }

    // ── Session context ───────────────────────────────────────────────

    /**
     * Wraps a TorchScriptRunner with session metadata.
     */
    public static class SessionContext implements AutoCloseable {
        private final TorchScriptRunner runner;
        private final long acquiredAt;
        private volatile long releasedAt;

        SessionContext(TorchScriptRunner runner) {
            this.runner = runner;
            this.acquiredAt = System.currentTimeMillis();
        }

        public TorchScriptRunner runner() {
            return runner;
        }

        public long acquiredAt() {
            return acquiredAt;
        }

        public long releasedAt() {
            return releasedAt;
        }

        void markReleased() {
            this.releasedAt = System.currentTimeMillis();
        }

        @Override
        public void close() {
            // Sessions are returned to pool, not closed directly
        }
    }

    // ── Internal pool ─────────────────────────────────────────────────

    private class SessionPool {
        private final Path modelPath;
        private final Device device;
        private final ConcurrentHashMap<SessionContext, Boolean> activeSessions = new ConcurrentHashMap<>();
        private final ConcurrentLinkedDeque<SessionContext> idleSessions = new ConcurrentLinkedDeque<>();
        private final AtomicInteger totalCreatedCounter = new AtomicInteger(0);
        private final Semaphore permits;

        SessionPool(Path modelPath, Device device) {
            this.modelPath = modelPath;
            this.device = device;
            // Use the smaller of per-tenant and global max as the permit count
            int maxPerTenant = config.session().maxPerTenant();
            this.permits = new Semaphore(maxPerTenant, /* fair */ true);
        }

        SessionContext acquire() {
            // 1. Acquire a permit with backpressure (wait up to timeout)
            int timeoutSec = config.inference().timeoutSeconds();
            try {
                if (!permits.tryAcquire(timeoutSec, TimeUnit.SECONDS)) {
                    throw new RuntimeException(
                            "Session pool exhausted for this tenant. Waited " + timeoutSec
                                    + "s. Max per-tenant sessions: " + config.session().maxPerTenant()
                                    + ". Consider increasing libtorch.provider.session.max-per-tenant.");
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Interrupted while waiting for session permit", e);
            }

            try {
                // 2. Try to reuse an idle session
                SessionContext idle;
                while ((idle = idleSessions.pollFirst()) != null) {
                    if (!idle.runner().isClosed()) {
                        activeSessions.put(idle, Boolean.TRUE);
                        log.debugf("Reused idle session (active=%d, idle=%d, permits=%d)",
                                activeSessions.size(), idleSessions.size(), permits.availablePermits());
                        return idle;
                    }
                    // Skip closed runners (shouldn't happen, but be safe)
                }

                // 3. Check global limit before creating
                int maxTotal = config.session().maxTotal();
                int globalTotal = LibTorchSessionManager.this.totalSessionCount();
                if (globalTotal >= maxTotal) {
                    permits.release(); // Give back the per-tenant permit
                    throw new RuntimeException(
                            "Global session pool exhausted. Max total sessions: " + maxTotal
                                    + ". Active: " + LibTorchSessionManager.this.activeSessionCount()
                                    + ", Idle: " + LibTorchSessionManager.this.idleSessionCount());
                }

                // 4. Create a new session
                TorchScriptRunner runner = TorchScriptRunner.load(modelPath, device);
                SessionContext ctx = new SessionContext(runner);
                activeSessions.put(ctx, Boolean.TRUE);
                totalCreatedCounter.incrementAndGet();
                log.debugf("Created new session (active=%d, idle=%d, total_created=%d, permits=%d)",
                        activeSessions.size(), idleSessions.size(),
                        totalCreatedCounter.get(), permits.availablePermits());
                return ctx;
            } catch (RuntimeException e) {
                // If session creation fails, release the permit
                if (!(e.getMessage() != null && e.getMessage().startsWith("Global session pool"))) {
                    permits.release();
                }
                throw e;
            }
        }

        void release(SessionContext session) {
            activeSessions.remove(session);
            permits.release(); // Return the permit for the next waiter
            if (!session.runner().isClosed()) {
                // Return to idle pool for reuse
                session.markReleased();
                idleSessions.addLast(session);
                log.debugf("Session returned to idle pool (active=%d, idle=%d, permits=%d)",
                        activeSessions.size(), idleSessions.size(), permits.availablePermits());
            }
        }

        /**
         * Evict sessions that have been idle since before the given timestamp.
         *
         * @return number of sessions evicted
         */
        int evictIdle(long idleThreshold) {
            int evicted = 0;
            var iter = idleSessions.iterator();
            while (iter.hasNext()) {
                SessionContext ctx = iter.next();
                if (ctx.releasedAt() > 0 && ctx.releasedAt() < idleThreshold) {
                    iter.remove();
                    closeRunner(ctx);
                    evicted++;
                }
            }
            return evicted;
        }

        int activeCount() {
            return activeSessions.size();
        }

        int idleCount() {
            return idleSessions.size();
        }

        int totalCreated() {
            return totalCreatedCounter.get();
        }

        void shutdown() {
            // Close all active sessions
            activeSessions.keySet().forEach(ctx -> {
                try {
                    ctx.runner().close();
                } catch (Exception e) {
                    log.warnf(e, "Error closing active session runner");
                }
            });
            activeSessions.clear();

            // Close all idle sessions
            SessionContext ctx;
            while ((ctx = idleSessions.pollFirst()) != null) {
                closeRunner(ctx);
            }
        }
    }
}
