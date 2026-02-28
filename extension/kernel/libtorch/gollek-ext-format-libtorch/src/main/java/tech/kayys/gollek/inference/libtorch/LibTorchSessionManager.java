package tech.kayys.gollek.inference.libtorch;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.core.Device;
import tech.kayys.gollek.provider.core.session.AdaptiveSessionEvictionPolicy;
import tech.kayys.gollek.provider.core.session.AdaptiveSessionEvictionState;
import tech.kayys.gollek.provider.core.session.EwmaAdaptiveSessionEvictionPolicy;
import tech.kayys.gollek.spi.inference.AdapterSpec;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

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
    private final Map<String, ReentrantLock> adapterInitLocks = new ConcurrentHashMap<>();
    private final Map<String, java.util.Set<String>> tenantAdapterPoolKeys = new ConcurrentHashMap<>();
    private final AdaptiveSessionEvictionState adaptiveEvictionState = new AdaptiveSessionEvictionState();
    private ScheduledExecutorService evictor;

    @Inject
    LibTorchProviderConfig config;

    @Inject
    LibTorchAdapterApplier adapterApplier;

    @Inject
    LibTorchMetrics metrics;
    @Inject
    AdaptiveSessionEvictionPolicy adaptiveEvictionPolicy;

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
        return getSession(tenantId, modelId, config, null);
    }

    public SessionContext getSession(String tenantId, String modelId, LibTorchProviderConfig config,
            AdapterSpec adapterSpec) {
        ModelSelection selection = resolveModelSelection(tenantId, modelId, config, adapterSpec);
        String poolKey = buildPoolKey(tenantId, modelId, adapterSpec);
        enforceTenantAdapterQuota(tenantId, poolKey, adapterSpec);
        SessionPool pool = pools.computeIfAbsent(poolKey,
                k -> new SessionPool(k, tenantId, selection.modelPath(), getDevice(), selection.runtimeLoraPath(),
                        adapterSpec));

        return pool.acquire();
    }

    /**
     * @deprecated Use {@link #getSession(String, String, LibTorchProviderConfig)}
     */
    @Deprecated
    public SessionContext acquire(String tenantId, String modelId, Path modelPath) {
        String poolKey = tenantId + ":" + modelId;
        SessionPool pool = pools.computeIfAbsent(poolKey,
                k -> new SessionPool(k, tenantId, modelPath, getDevice(), null, null));
        return pool.acquire();
    }

    /**
     * Release a session back to the pool.
     */
    public void releaseSession(String tenantId, String modelId, SessionContext session) {
        releaseSession(tenantId, modelId, session, null);
    }

    public void releaseSession(String tenantId, String modelId, SessionContext session, AdapterSpec adapterSpec) {
        String poolKey = buildPoolKey(tenantId, modelId, adapterSpec);
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
        return resolveModelPath("community", modelId, config, null);
    }

    public Path resolveModelPath(String modelId, LibTorchProviderConfig config, AdapterSpec adapterSpec) {
        return resolveModelPath("community", modelId, config, adapterSpec);
    }

    public Path resolveModelPath(String tenantId, String modelId, LibTorchProviderConfig config, AdapterSpec adapterSpec) {
        return resolveModelSelection(tenantId, modelId, config, adapterSpec).modelPath();
    }

    private ModelSelection resolveModelSelection(String tenantId, String modelId, LibTorchProviderConfig config,
            AdapterSpec adapterSpec) {
        Path baseModelPath = resolveBaseModelPath(modelId, config);
        if (adapterSpec == null) {
            return new ModelSelection(baseModelPath, null);
        }

        Path adapterPath = resolveAdapterPath(adapterSpec, config, tenantId);
        if (isRuntimeLoraArtifact(adapterPath)) {
            return new ModelSelection(baseModelPath, adapterPath);
        }

        if (!config.adapter().allowPrecompiledModelPath()) {
            throw new UnsupportedOperationException(
                    "Adapter path '" + adapterPath + "' requires precompiled-model routing, but "
                            + "libtorch.provider.adapter.allow-precompiled-model-path=false.");
        }
        if (!isPrecompiledModelArtifact(adapterPath)) {
            throw new UnsupportedOperationException(
                    "Unsupported LibTorch adapter artifact: " + adapterPath
                            + ". Expected runtime LoRA (.safetensors/.safetensor) or precompiled model (.pt/.pts/.pth/.bin).");
        }
        return new ModelSelection(adapterPath, null);
    }

    private Path resolveBaseModelPath(String modelId, LibTorchProviderConfig config) {
        String basePath = config.model().basePath();
        String extensions = config.model().extensions();

        for (String ext : extensions.split(",")) {
            Path path = Path.of(basePath, modelId + ext.trim());
            if (Files.exists(path)) {
                return path;
            }
        }

        // Fallback for absolute paths or already-extensioned IDs
        Path directPath = Path.of(modelId);
        if (Files.exists(directPath)) {
            return directPath;
        }

        throw new RuntimeException("Model not found: " + modelId + " in " + basePath);
    }

    private Path resolveAdapterPath(AdapterSpec adapterSpec, LibTorchProviderConfig config, String tenantId) {
        if (adapterSpec == null) {
            throw new IllegalArgumentException("adapterSpec cannot be null");
        }
        if (!config.adapter().enabled()) {
            throw new IllegalArgumentException("Adapters are disabled for LibTorch provider");
        }
        if (!adapterSpec.isType("lora")) {
            throw new IllegalArgumentException(
                    "Unsupported adapter_type for LibTorch: " + adapterSpec.type() + " (expected: lora)");
        }
        String rawPath = adapterSpec.adapterPath();
        if (rawPath == null || rawPath.isBlank()) {
            throw new IllegalArgumentException("adapter_path is required for LibTorch adapter routing");
        }

        Path adapterPath = Path.of(rawPath);
        if (!adapterPath.isAbsolute()) {
            adapterPath = Path.of(config.adapter().basePath()).resolve(adapterPath).normalize();
        }
        if (!Files.exists(adapterPath)) {
            throw new IllegalArgumentException("Adapter path not found: " + adapterPath);
        }
        enforceRolloutPolicy(normalizeTenantId(tenantId), adapterSpec, adapterPath, config);
        return adapterPath;
    }

    private boolean isRuntimeLoraArtifact(Path adapterPath) {
        String name = adapterPath.getFileName().toString().toLowerCase();
        return name.endsWith(".safetensors") || name.endsWith(".safetensor");
    }

    private boolean isPrecompiledModelArtifact(Path adapterPath) {
        String name = adapterPath.getFileName().toString().toLowerCase();
        return name.endsWith(".pt")
                || name.endsWith(".pts")
                || name.endsWith(".pth")
                || name.endsWith(".bin");
    }

    private String buildPoolKey(String tenantId, String modelId, AdapterSpec adapterSpec) {
        String adapterKey = adapterSpec == null ? "no-adapter" : adapterSpec.cacheKey();
        return tenantId + ":" + modelId + ":" + adapterKey;
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
        int maxTotal = Math.max(1, config.session().maxTotal());
        double utilization = (double) totalSessionCount() / maxTotal;
        long idleThreshold = System.currentTimeMillis()
                - (adaptiveIdleTimeoutSeconds() * 1000L);
        int evictedTotal = 0;
        for (var entry : pools.entrySet()) {
            SessionPool pool = entry.getValue();
            int evicted = pool.evictIdle(idleThreshold);
            evictedTotal += evicted;
            if (evicted > 0) {
                log.debugf("Evicted %d idle sessions from pool %s", evicted, entry.getKey());
            }
            if (pool.isDrained() && pools.remove(entry.getKey(), pool)) {
                deregisterTenantAdapterPool(pool);
            }
        }
        recordAdaptiveTelemetry(utilization >= 0.75d, evictedTotal);
    }

    private int evictIdleSessionsUnderPressure() {
        long aggressiveThreshold = System.currentTimeMillis()
                - (Math.max(5, config.session().idleTimeoutSeconds() / 4) * 1000L);
        int evictedTotal = 0;
        for (var entry : pools.entrySet()) {
            SessionPool pool = entry.getValue();
            int evicted = pool.evictIdle(aggressiveThreshold);
            evictedTotal += evicted;
            if (pool.isDrained() && pools.remove(entry.getKey(), pool)) {
                deregisterTenantAdapterPool(pool);
            }
        }
        recordAdaptiveTelemetry(true, evictedTotal);
        return evictedTotal;
    }

    int adaptiveIdleTimeoutSeconds() {
        int base = Math.max(1, config.session().idleTimeoutSeconds());
        int maxTotal = Math.max(1, config.session().maxTotal());
        double utilization = (double) totalSessionCount() / maxTotal;
        int resolved = policy().resolveIdleTimeoutSeconds(adaptiveEvictionState, base, utilization);
        if (metrics != null) {
            metrics.recordSessionEvictionTelemetry(adaptivePressureScore(), resolved, 0);
        }
        return resolved;
    }

    void recordAdaptiveTelemetryForTest(boolean underPressure, int reclaimedSessions) {
        recordAdaptiveTelemetry(underPressure, reclaimedSessions);
    }

    double adaptivePressureScoreForTest() {
        return adaptivePressureScore();
    }

    private void recordAdaptiveTelemetry(boolean underPressure, int reclaimedSessions) {
        policy().recordTelemetry(adaptiveEvictionState, underPressure, reclaimedSessions);
        if (metrics != null) {
            metrics.recordSessionEvictionTelemetry(adaptivePressureScore(), adaptiveIdleTimeoutSeconds(), reclaimedSessions);
        }
    }

    private double adaptivePressureScore() {
        return policy().pressureScore(adaptiveEvictionState);
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
        tenantAdapterPoolKeys.clear();
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

    void enforceTenantAdapterQuota(String tenantId, String poolKey, AdapterSpec adapterSpec) {
        if (adapterSpec == null) {
            return;
        }
        synchronized (tenantAdapterPoolKeys) {
            if (pools.containsKey(poolKey)) {
                return;
            }
            java.util.Set<String> poolsForTenant = tenantAdapterPoolKeys.computeIfAbsent(
                    normalizeTenantId(tenantId), __ -> ConcurrentHashMap.newKeySet());
            if (poolsForTenant.contains(poolKey)) {
                return;
            }

            int configured = config.adapter().maxActivePoolsPerTenant();
            int maxPerTenant = configured > 0
                    ? configured
                    : Math.max(1, config.session().maxPerTenant());
            if (poolsForTenant.size() >= maxPerTenant) {
                throw new RuntimeException(
                        "Adapter pool quota exceeded for tenant '" + normalizeTenantId(tenantId) + "': max="
                                + maxPerTenant
                                + " unique adapter pools. Increase libtorch.provider.session.max-per-tenant or reuse adapters.");
            }
            poolsForTenant.add(poolKey);
        }
    }

    private void deregisterTenantAdapterPool(SessionPool pool) {
        if (!pool.hasAdapter()) {
            return;
        }
        synchronized (tenantAdapterPoolKeys) {
            java.util.Set<String> poolsForTenant = tenantAdapterPoolKeys.get(pool.tenantId());
            if (poolsForTenant == null) {
                return;
            }
            poolsForTenant.remove(pool.poolKey());
            if (poolsForTenant.isEmpty()) {
                tenantAdapterPoolKeys.remove(pool.tenantId(), poolsForTenant);
            }
        }
    }

    private String normalizeTenantId(String tenantId) {
        if (tenantId == null || tenantId.isBlank()) {
            return "community";
        }
        return tenantId;
    }

    private AdaptiveSessionEvictionPolicy policy() {
        return adaptiveEvictionPolicy != null ? adaptiveEvictionPolicy : EwmaAdaptiveSessionEvictionPolicy.DEFAULT;
    }

    private void enforceRolloutPolicy(String tenantId, AdapterSpec adapterSpec, Path adapterPath,
            LibTorchProviderConfig config) {
        if (!config.adapter().rolloutGuardEnabled()) {
            return;
        }
        if (isBlockedTenant(tenantId, config)) {
            throw new IllegalArgumentException("Tenant '" + tenantId + "' is not allowed for adapter rollout");
        }
        if (isBlockedAdapterId(adapterSpec.adapterId(), config)) {
            throw new IllegalArgumentException(
                    "Adapter id '" + adapterSpec.adapterId() + "' is blocked by rollout policy");
        }
        if (isBlockedPath(adapterPath, config)) {
            throw new IllegalArgumentException(
                    "Adapter path '" + adapterPath + "' is blocked by rollout policy");
        }
    }

    private boolean isBlockedTenant(String tenantId, LibTorchProviderConfig config) {
        var allowed = config.adapter().rolloutAllowedTenants().orElse(java.util.List.of());
        if (allowed.isEmpty()) {
            return false;
        }
        return allowed.stream()
                .filter(v -> v != null && !v.isBlank())
                .noneMatch(v -> v.equalsIgnoreCase(tenantId));
    }

    private boolean isBlockedAdapterId(String adapterId, LibTorchProviderConfig config) {
        if (adapterId == null || adapterId.isBlank()) {
            return false;
        }
        var blocked = config.adapter().rolloutBlockedAdapterIds().orElse(java.util.List.of());
        return blocked.stream()
                .filter(v -> v != null && !v.isBlank())
                .anyMatch(v -> v.equalsIgnoreCase(adapterId));
    }

    private boolean isBlockedPath(Path adapterPath, LibTorchProviderConfig config) {
        if (adapterPath == null) {
            return false;
        }
        String normalized = adapterPath.toString().toLowerCase(Locale.ROOT);
        var blockedPrefixes = config.adapter().rolloutBlockedPathPrefixes().orElse(java.util.List.of());
        return blockedPrefixes.stream()
                .filter(v -> v != null && !v.isBlank())
                .map(v -> v.toLowerCase(Locale.ROOT))
                .anyMatch(normalized::startsWith);
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
        private final String poolKey;
        private final String tenantId;
        private final Path modelPath;
        private final Device device;
        private final Path runtimeLoraPath;
        private final AdapterSpec adapterSpec;
        private final ConcurrentHashMap<SessionContext, Boolean> activeSessions = new ConcurrentHashMap<>();
        private final ConcurrentLinkedDeque<SessionContext> idleSessions = new ConcurrentLinkedDeque<>();
        private final AtomicInteger totalCreatedCounter = new AtomicInteger(0);
        private final Semaphore permits;

        SessionPool(String poolKey, String tenantId, Path modelPath, Device device, Path runtimeLoraPath,
                AdapterSpec adapterSpec) {
            this.poolKey = poolKey;
            this.tenantId = normalizeTenantId(tenantId);
            this.modelPath = modelPath;
            this.device = device;
            this.runtimeLoraPath = runtimeLoraPath;
            this.adapterSpec = adapterSpec;
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
                    int reclaimed = LibTorchSessionManager.this.evictIdleSessionsUnderPressure();
                    globalTotal = LibTorchSessionManager.this.totalSessionCount();
                    if (globalTotal >= maxTotal) {
                        permits.release(); // Give back the per-tenant permit
                        throw new RuntimeException(
                                "Global session pool exhausted. Max total sessions: " + maxTotal
                                        + ". Active: " + LibTorchSessionManager.this.activeSessionCount()
                                        + ", Idle: " + LibTorchSessionManager.this.idleSessionCount()
                                        + ", Reclaimed: " + reclaimed);
                    }
                }

                // 4. Create a new session (guarded for adapter cold-start stampede)
                return createSessionWithGuard();
            } catch (RuntimeException e) {
                // If session creation fails, release the permit
                if (!(e.getMessage() != null && e.getMessage().startsWith("Global session pool"))) {
                    permits.release();
                }
                throw e;
            }
        }

        private SessionContext createSessionWithGuard() {
            String adapterKey = adapterSpec != null ? adapterSpec.cacheKey() : null;
            ReentrantLock initLock = adapterKey == null
                    ? null
                    : adapterInitLocks.computeIfAbsent(adapterKey, __ -> new ReentrantLock(true));

            if (initLock != null) {
                long waitStartNanos = System.nanoTime();
                initLock.lock();
                if (metrics != null) {
                    metrics.recordAdapterInitWait(Duration.ofNanos(System.nanoTime() - waitStartNanos));
                }
            }
            try {
                // Double-check idle after waiting on the adapter init lock.
                SessionContext idle = pollIdleSession();
                if (idle != null) {
                    activeSessions.put(idle, Boolean.TRUE);
                    log.debugf("Reused idle session after adapter init lock (active=%d, idle=%d, permits=%d)",
                            activeSessions.size(), idleSessions.size(), permits.availablePermits());
                    return idle;
                }

                TorchScriptRunner runner = TorchScriptRunner.load(modelPath, device);
                if (runtimeLoraPath != null) {
                    float scale = adapterSpec != null ? adapterSpec.scale() : 1.0f;
                    long applyStartNanos = System.nanoTime();
                    int applied = adapterApplier.applyRuntimeLora(runner, runtimeLoraPath, scale);
                    if (metrics != null) {
                        metrics.recordAdapterApply(Duration.ofNanos(System.nanoTime() - applyStartNanos));
                    }
                    log.debugf("Applied runtime LoRA adapter (path=%s, updates=%d, scale=%.4f)",
                            runtimeLoraPath.getFileName(), applied, scale);
                }

                SessionContext ctx = new SessionContext(runner);
                activeSessions.put(ctx, Boolean.TRUE);
                totalCreatedCounter.incrementAndGet();
                log.debugf("Created new session (active=%d, idle=%d, total_created=%d, permits=%d)",
                        activeSessions.size(), idleSessions.size(),
                        totalCreatedCounter.get(), permits.availablePermits());
                return ctx;
            } finally {
                if (initLock != null) {
                    initLock.unlock();
                }
            }
        }

        private SessionContext pollIdleSession() {
            SessionContext idle;
            while ((idle = idleSessions.pollFirst()) != null) {
                if (!idle.runner().isClosed()) {
                    return idle;
                }
            }
            return null;
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

        boolean isDrained() {
            return activeSessions.isEmpty() && idleSessions.isEmpty();
        }

        boolean hasAdapter() {
            return adapterSpec != null;
        }

        String tenantId() {
            return tenantId;
        }

        String poolKey() {
            return poolKey;
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

    private record ModelSelection(Path modelPath, Path runtimeLoraPath) {
    }
}
