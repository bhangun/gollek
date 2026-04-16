package tech.kayys.gollek.provider.litert;

import org.jboss.logging.Logger;
import tech.kayys.gollek.provider.core.session.AdaptiveSessionEvictionPolicy;
import tech.kayys.gollek.provider.core.session.AdaptiveSessionEvictionState;
import tech.kayys.gollek.provider.core.session.EwmaAdaptiveSessionEvictionPolicy;

import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Session Manager for LiteRT 2.0.
 *
 * <p>Manages pools of LiteRT runners for efficient reuse.
 * This class has been decoupled from Micrometer and Jakarta CDI to support
 * standalone use cases (JBang/FFI).
 */
@jakarta.enterprise.context.ApplicationScoped
public class LiteRTSessionManager {

    private static final Logger LOG = Logger.getLogger(LiteRTSessionManager.class);

    private final Map<String, SessionPool> pools = new ConcurrentHashMap<>();
    private final AdaptiveSessionEvictionState adaptiveEvictionState = new AdaptiveSessionEvictionState();
    private final AtomicLong adaptiveIdleTimeoutSeconds = new AtomicLong(300);
    private final AtomicLong adaptivePressureScorePermille = new AtomicLong(0);

    private volatile boolean adaptiveMetricsRegistered;
    private ScheduledExecutorService evictor;

    @jakarta.inject.Inject
    LiteRTProviderConfig config;
    
    private final AdaptiveSessionEvictionPolicy adaptiveEvictionPolicy;

    /**
     * Default constructor for CDI.
     */
    public LiteRTSessionManager() {
        this.adaptiveEvictionPolicy = EwmaAdaptiveSessionEvictionPolicy.DEFAULT;
    }

    /**
     * Constructor for manual instantiation (Standalone/JBang).
     */
    public LiteRTSessionManager(LiteRTProviderConfig config) {
        this(config, EwmaAdaptiveSessionEvictionPolicy.DEFAULT);
    }

    public LiteRTSessionManager(LiteRTProviderConfig config, AdaptiveSessionEvictionPolicy policy) {
        this.config = config;
        this.adaptiveEvictionPolicy = policy;
    }

    public SessionContext getSession(String tenantId, String modelId, Path modelPath, LiteRTRunnerConfig runnerConfig) {
        return getSession(tenantId, modelId, modelPath, runnerConfig, this::createRunner);
    }

    SessionContext getSession(String tenantId, String modelId, Path modelPath, LiteRTRunnerConfig runnerConfig,
            RunnerFactory runnerFactory) {
        String key = poolKey(tenantId, modelId);
        SessionPool pool = pools.computeIfAbsent(key, __ -> new SessionPool(key, tenantId, modelPath, runnerConfig));
        return pool.acquire(runnerFactory);
    }

    public void releaseSession(String tenantId, String modelId, SessionContext sessionContext) {
        if (sessionContext == null) {
            return;
        }
        String key = poolKey(tenantId, modelId);
        SessionPool pool = pools.get(key);
        if (pool != null) {
            pool.release(sessionContext);
        } else {
            closeRunner(sessionContext.runner());
        }
    }

    public void startEvictor() {
        if (evictor != null) {
            return;
        }
        int idleTimeout = Math.max(1, config.session().idleTimeoutSeconds());
        long intervalSeconds = Math.max(10, idleTimeout / 2L);
        evictor = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "gollek-litert-session-evictor");
            t.setDaemon(true);
            return t;
        });
        evictor.scheduleAtFixedRate(this::evictIdleSessions, intervalSeconds, intervalSeconds, TimeUnit.SECONDS);
        LOG.info("LiteRT session evictor started");
    }

    public void shutdown() {
        if (evictor != null) {
            evictor.shutdownNow();
            evictor = null;
        }
        pools.values().forEach(SessionPool::shutdown);
        pools.clear();
        LOG.info("LiteRT session manager shutdown complete");
    }

    int adaptiveIdleTimeoutSeconds() {
        int base = Math.max(1, config.session().idleTimeoutSeconds());
        double utilization = (double) totalSessionCount() / Math.max(1, config.session().maxTotal());
        int resolved = policy().resolveIdleTimeoutSeconds(adaptiveEvictionState, base, utilization);
        adaptiveIdleTimeoutSeconds.set(resolved);
        return resolved;
    }

    private void evictIdleSessions() {
        long threshold = System.currentTimeMillis() - (adaptiveIdleTimeoutSeconds() * 1000L);
        int evictedTotal = 0;
        double utilization = (double) totalSessionCount() / Math.max(1, config.session().maxTotal());

        for (var entry : pools.entrySet()) {
            SessionPool pool = entry.getValue();
            int evicted = pool.evictIdle(threshold);
            evictedTotal += evicted;
            if (pool.isDrained() && pools.remove(entry.getKey(), pool)) {
                LOG.debugf("Removed drained LiteRT pool %s", entry.getKey());
            }
        }
        recordAdaptiveTelemetry(utilization >= 0.75d, evictedTotal);
    }

    private int evictIdleSessionsUnderPressure() {
        long threshold = System.currentTimeMillis() - (Math.max(5, config.session().idleTimeoutSeconds() / 4) * 1000L);
        int evictedTotal = 0;
        for (var entry : pools.entrySet()) {
            SessionPool pool = entry.getValue();
            int evicted = pool.evictIdle(threshold);
            evictedTotal += evicted;
            if (pool.isDrained() && pools.remove(entry.getKey(), pool)) {
                LOG.debugf("Removed drained LiteRT pool %s under pressure", entry.getKey());
            }
        }
        recordAdaptiveTelemetry(true, evictedTotal);
        return evictedTotal;
    }

    private void recordAdaptiveTelemetry(boolean underPressure, int reclaimedSessions) {
        policy().recordTelemetry(adaptiveEvictionState, underPressure, reclaimedSessions);
        double score = adaptivePressureScore();
        adaptivePressureScorePermille.set(Math.round(score * 1000.0d));
        // Note: Micrometer metrics removed for standalone stability
    }

    private double adaptivePressureScore() {
        return policy().pressureScore(adaptiveEvictionState);
    }

    private AdaptiveSessionEvictionPolicy policy() {
        return adaptiveEvictionPolicy != null ? adaptiveEvictionPolicy : EwmaAdaptiveSessionEvictionPolicy.DEFAULT;
    }

    // ====================================================================
    // TEST HELPERS (Package-Private)
    // ====================================================================

    void recordAdaptiveTelemetryForTest(boolean underPressure, int reclaimedSessions) {
        recordAdaptiveTelemetry(underPressure, reclaimedSessions);
    }

    double adaptivePressureScoreForTest() {
        return adaptivePressureScore();
    }

    private int totalSessionCount() {
        return pools.values().stream().mapToInt(SessionPool::totalCount).sum();
    }

    private String poolKey(String tenantId, String modelId) {
        String normalizedTenant = (tenantId == null || tenantId.isBlank()) ? "community" : tenantId;
        return normalizedTenant + ":" + modelId;
    }

    private LiteRTCpuRunner createRunner(Path modelPath, LiteRTRunnerConfig runnerConfig) {
        LiteRTCpuRunner runner = new LiteRTCpuRunner();
        runner.initialize(modelPath, runnerConfig);
        return runner;
    }

    private void closeRunner(LiteRTCpuRunner runner) {
        if (runner == null) {
            return;
        }
        try {
            runner.close();
        } catch (Exception e) {
            LOG.warnf(e, "Failed to close LiteRT runner");
        }
    }

    public void registerAdaptiveMetrics(Object meterRegistry) {
        // No-op: Decoupled from Micrometer to support standalone mode.
        // If metrics are needed, use a wrapper/SPI bridge.
    }

    interface RunnerFactory {
        LiteRTCpuRunner create(Path modelPath, LiteRTRunnerConfig runnerConfig);
    }

    public static final class SessionContext {
        private final LiteRTCpuRunner runner;
        private final long acquiredAt;
        private volatile long releasedAt;

        SessionContext(LiteRTCpuRunner runner) {
            this.runner = runner;
            this.acquiredAt = System.currentTimeMillis();
        }

        public LiteRTCpuRunner runner() {
            return runner;
        }

        public long releasedAt() {
            return releasedAt;
        }

        void markReleased() {
            releasedAt = System.currentTimeMillis();
        }
    }

    private final class SessionPool {
        private final String poolKey;
        private final String tenantId;
        private final Path modelPath;
        private final LiteRTRunnerConfig runnerConfig;
        private final ConcurrentHashMap<SessionContext, Boolean> active = new ConcurrentHashMap<>();
        private final ConcurrentLinkedDeque<SessionContext> idle = new ConcurrentLinkedDeque<>();
        private final Semaphore permits;

        private SessionPool(String poolKey, String tenantId, Path modelPath, LiteRTRunnerConfig runnerConfig) {
            this.poolKey = poolKey;
            this.tenantId = tenantId;
            this.modelPath = modelPath;
            this.runnerConfig = runnerConfig;
            this.permits = new Semaphore(Math.max(1, config.session().maxPerTenant()), true);
        }

        SessionContext acquire(RunnerFactory factory) {
            long timeoutSeconds = Math.max(1L, config.defaultTimeout().toSeconds());
            try {
                if (!permits.tryAcquire(timeoutSeconds, TimeUnit.SECONDS)) {
                    throw new RuntimeException("LiteRT session pool exhausted for tenant '" + tenantId + "'");
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Interrupted acquiring LiteRT session", e);
            }

            try {
                SessionContext reused = pollIdle();
                if (reused != null) {
                    active.put(reused, Boolean.TRUE);
                    return reused;
                }
                if (totalSessionCount() >= Math.max(1, config.session().maxTotal())) {
                    int reclaimed = evictIdleSessionsUnderPressure();
                    if (totalSessionCount() >= Math.max(1, config.session().maxTotal())) {
                        permits.release();
                        throw new RuntimeException(
                                "LiteRT global session pool exhausted (reclaimed=" + reclaimed + ")");
                    }
                }
                SessionContext created = new SessionContext(factory.create(modelPath, runnerConfig));
                active.put(created, Boolean.TRUE);
                return created;
            } catch (RuntimeException e) {
                if (!(e.getMessage() != null && e.getMessage().startsWith("LiteRT global session pool"))) {
                    permits.release();
                }
                throw e;
            }
        }

        void release(SessionContext sessionContext) {
            active.remove(sessionContext);
            sessionContext.markReleased();
            idle.addLast(sessionContext);
            permits.release();
        }

        int evictIdle(long idleThreshold) {
            int evicted = 0;
            var iter = idle.iterator();
            while (iter.hasNext()) {
                SessionContext ctx = iter.next();
                if (ctx.releasedAt() > 0 && ctx.releasedAt() < idleThreshold) {
                    iter.remove();
                    closeRunner(ctx.runner());
                    evicted++;
                }
            }
            return evicted;
        }

        SessionContext pollIdle() {
            SessionContext ctx = idle.pollFirst();
            if (ctx != null && !ctx.runner().health()) {
                closeRunner(ctx.runner());
                return pollIdle();
            }
            return ctx;
        }

        int totalCount() {
            return active.size() + idle.size();
        }

        boolean isDrained() {
            return active.isEmpty() && idle.isEmpty();
        }

        void shutdown() {
            active.keySet().forEach(context -> closeRunner(context.runner()));
            active.clear();
            SessionContext ctx;
            while ((ctx = idle.pollFirst()) != null) {
                closeRunner(ctx.runner());
            }
        }
    }
}
