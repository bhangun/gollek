package tech.kayys.gollek.multimodal;

import io.quarkus.runtime.StartupEvent;
import io.quarkus.runtime.ShutdownEvent;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.net.http.HttpClient;
import java.time.Duration;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Adaptive connection pool for provider HTTP clients.
 *
 * <p>
 * Each provider gets its own {@link HttpClient} instance backed by a
 * dedicated virtual-thread executor (Java 21+) or a fixed thread pool on
 * older JVMs. Pool sizes auto-scale based on rolling latency P99 —
 * if P99 exceeds the target, the pool grows; if it stays well below, it
 * shrinks.
 *
 * <h3>Why per-provider pools?</h3>
 * <ul>
 * <li>Isolates slow providers from fast ones (bulkhead pattern).</li>
 * <li>Enables per-provider connection limits (Anthropic = 5 RPS, Gemini = 60
 * RPS).</li>
 * <li>Allows fine-grained back-pressure without a global semaphore.</li>
 * </ul>
 *
 * <h3>Config</h3>
 * 
 * <pre>
 *   gollek.pool.default.max-connections=50
 *   gollek.pool.default.connect-timeout-ms=5000
 *   gollek.pool.default.request-timeout-ms=30000
 *   gollek.pool.adaptive.enabled=true
 *   gollek.pool.adaptive.target-p99-ms=500
 *   gollek.pool.adaptive.min-connections=5
 *   gollek.pool.adaptive.max-connections=200
 *   gollek.pool.adaptive.scale-interval-secs=30
 * </pre>
 */
@ApplicationScoped
public class ProviderConnectionPool {

    private static final Logger LOG = Logger.getLogger(ProviderConnectionPool.class);

    @ConfigProperty(name = "gollek.pool.default.max-connections", defaultValue = "50")
    int defaultMaxConn;

    @ConfigProperty(name = "gollek.pool.default.connect-timeout-ms", defaultValue = "5000")
    int connectTimeoutMs;

    @ConfigProperty(name = "gollek.pool.adaptive.enabled", defaultValue = "true")
    boolean adaptiveEnabled;

    @ConfigProperty(name = "gollek.pool.adaptive.target-p99-ms", defaultValue = "500")
    long targetP99Ms;

    @ConfigProperty(name = "gollek.pool.adaptive.min-connections", defaultValue = "5")
    int minConn;

    @ConfigProperty(name = "gollek.pool.adaptive.max-connections", defaultValue = "200")
    int maxConn;

    @ConfigProperty(name = "gollek.pool.adaptive.scale-interval-secs", defaultValue = "30")
    int scaleIntervalSecs;

    // providerId → pool entry
    private final ConcurrentHashMap<String, PoolEntry> pools = new ConcurrentHashMap<>();
    private ScheduledExecutorService scaler;

    void onStart(@Observes StartupEvent ev) {
        scaler = Executors.newSingleThreadScheduledExecutor(
                r -> Thread.ofVirtual().name("pool-scaler").unstarted(r));
        if (adaptiveEnabled) {
            scaler.scheduleAtFixedRate(this::adaptPools,
                    scaleIntervalSecs, scaleIntervalSecs, TimeUnit.SECONDS);
        }
        LOG.info("Provider connection pool manager started (adaptive=" + adaptiveEnabled + ")");
    }

    @jakarta.annotation.PreDestroy
    public void shutdown() {
        scaler.shutdownNow();
        pools.values().forEach(PoolEntry::close);
        LOG.info("Provider connection pool manager stopped");
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Acquire the {@link HttpClient} for a given provider, creating it lazily.
     */
    public HttpClient clientFor(String providerId) {
        return pools.computeIfAbsent(providerId, this::createEntry).client();
    }

    /**
     * Acquire a permit before sending a request (back-pressure).
     * Returns false immediately if the pool is saturated (caller should shed load).
     */
    public boolean tryAcquire(String providerId) {
        PoolEntry entry = pools.computeIfAbsent(providerId, this::createEntry);
        return entry.semaphore().tryAcquire();
    }

    /** Release a permit after the request completes. */
    public void release(String providerId) {
        PoolEntry entry = pools.get(providerId);
        if (entry != null)
            entry.semaphore().release();
    }

    /**
     * Record a completed request latency for adaptive scaling.
     *
     * @param providerId provider that served the request
     * @param latencyMs  end-to-end latency in milliseconds
     */
    public void recordLatency(String providerId, long latencyMs) {
        PoolEntry entry = pools.get(providerId);
        if (entry != null)
            entry.recordLatency(latencyMs);
    }

    /** Returns a snapshot of pool stats for all providers. */
    public Map<String, PoolStats> stats() {
        var result = new java.util.LinkedHashMap<String, PoolStats>();
        pools.forEach((id, entry) -> result.put(id, entry.stats()));
        return result;
    }

    // -------------------------------------------------------------------------
    // Adaptive scaling
    // -------------------------------------------------------------------------

    private void adaptPools() {
        pools.forEach((id, entry) -> {
            long p99 = entry.p99Ms();
            int current = entry.semaphore().availablePermits()
                    + (entry.currentSize() - entry.semaphore().availablePermits());
            int current2 = entry.currentSize();

            if (p99 > targetP99Ms * 1.5 && current2 < maxConn) {
                // Latency too high → grow the pool
                int newSize = Math.min(current2 + current2 / 4 + 2, maxConn);
                entry.resize(newSize);
                LOG.infof("[POOL-SCALE-UP] %s: %d→%d (p99=%dms)", id, current2, newSize, p99);
            } else if (p99 < targetP99Ms * 0.5 && current2 > minConn) {
                // Latency well within SLO → shrink to reclaim resources
                int newSize = Math.max(current2 - current2 / 5, minConn);
                entry.resize(newSize);
                LOG.infof("[POOL-SCALE-DOWN] %s: %d→%d (p99=%dms)", id, current2, newSize, p99);
            }
        });
    }

    // -------------------------------------------------------------------------
    // Pool entry construction
    // -------------------------------------------------------------------------

    private PoolEntry createEntry(String providerId) {
        // Virtual-thread executor per provider (Java 21+)
        ExecutorService executor;
        try {
            executor = (ExecutorService) Executors.class
                    .getMethod("newVirtualThreadPerTaskExecutor")
                    .invoke(null);
        } catch (Exception e) {
            // Fallback for Java < 21
            executor = Executors.newFixedThreadPool(defaultMaxConn,
                    r -> new Thread(r, "pool-" + providerId));
        }

        HttpClient client = HttpClient.newBuilder()
                .executor(executor)
                .connectTimeout(Duration.ofMillis(connectTimeoutMs))
                .version(HttpClient.Version.HTTP_2) // H/2 multiplexing
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();

        LOG.infof("[POOL-CREATE] %s (maxConn=%d)", providerId, defaultMaxConn);
        return new PoolEntry(providerId, client, defaultMaxConn, executor);
    }

    // =========================================================================
    // Inner types
    // =========================================================================

    static final class PoolEntry {
        private final String providerId;
        private volatile HttpClient client;
        private volatile Semaphore semaphore;
        private volatile int size;
        private final ExecutorService executor;

        // Latency ring buffer for P99 calculation (last 1000 samples)
        private static final int RING_SIZE = 1000;
        private final long[] latencyRing = new long[RING_SIZE];
        private final AtomicInteger ringIdx = new AtomicInteger(0);
        private final AtomicLong totalRequests = new AtomicLong(0);

        PoolEntry(String id, HttpClient client, int size, ExecutorService executor) {
            this.providerId = id;
            this.client = client;
            this.size = size;
            this.semaphore = new Semaphore(size, true);
            this.executor = executor;
        }

        HttpClient client() {
            return client;
        }

        Semaphore semaphore() {
            return semaphore;
        }

        int currentSize() {
            return size;
        }

        void recordLatency(long ms) {
            int idx = ringIdx.getAndIncrement() % RING_SIZE;
            latencyRing[idx] = ms;
            totalRequests.incrementAndGet();
        }

        long p99Ms() {
            long[] copy = latencyRing.clone();
            Arrays.sort(copy);
            return copy[(int) (RING_SIZE * 0.99)];
        }

        void resize(int newSize) {
            int delta = newSize - size;
            if (delta > 0) {
                semaphore.release(delta);
            } else if (delta < 0) {
                // Drain permits; don't block — just update the counter
                semaphore.tryAcquire(Math.min(-delta, semaphore.availablePermits()));
            }
            size = newSize;
        }

        PoolStats stats() {
            return new PoolStats(providerId, size,
                    semaphore.availablePermits(),
                    size - semaphore.availablePermits(),
                    totalRequests.get(), p99Ms());
        }

        void close() {
            executor.shutdownNow();
        }
    }

    public record PoolStats(String providerId, int poolSize,
            int availablePermits, int activeRequests,
            long totalRequests, long p99Ms) {
    }
}
