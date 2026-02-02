package tech.kayys.golek.engine.inference;

import io.micrometer.core.instrument.*;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import lombok.extern.slf4j.Slf4j;

import java.time.Duration;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Comprehensive metrics collection for inference operations.
 * 
 * <p>
 * Metrics Categories:
 * <ul>
 * <li>Request metrics (total, success, failure rates)</li>
 * <li>Latency metrics (P50, P95, P99)</li>
 * <li>Throughput metrics (requests per second)</li>
 * <li>Resource metrics (memory, GPU, concurrent requests)</li>
 * <li>Runner metrics (per-runner performance)</li>
 * <li>Tenant metrics (per-tenant usage)</li>
 * </ul>
 * 
 * <p>
 * All metrics are exported to Prometheus via Micrometer.
 * 
 * @author bhangun
 * @since 1.0.0
 */
@ApplicationScoped
@Slf4j
public class InferenceMetrics {

    @Inject
    MeterRegistry registry;

    // Active request tracking
    private final ConcurrentHashMap<String, AtomicLong> activeRequests = new ConcurrentHashMap<>();

    // Metric names
    private static final String REQUESTS_TOTAL = "inference.requests.total";
    private static final String REQUESTS_SUCCESS = "inference.requests.success";
    private static final String REQUESTS_FAILURE = "inference.requests.failure";
    private static final String LATENCY = "inference.latency";
    private static final String THROUGHPUT = "inference.throughput";
    private static final String QUEUE_SIZE = "inference.queue.size";
    private static final String ACTIVE_REQUESTS = "inference.requests.active";
    private static final String INPUT_SIZE = "inference.input.bytes";
    private static final String OUTPUT_SIZE = "inference.output.bytes";
    private static final String RUNNER_HEALTH = "inference.runner.health";
    private static final String QUOTA_USAGE = "inference.quota.usage";
    private static final String QUOTA_LIMIT = "inference.quota.limit";

    /**
     * Record successful inference request.
     */
    public void recordSuccess(String tenantId, String modelId, String runnerName, long latencyMs) {
        Tags tags = Tags.of(
                "tenant", tenantId,
                "model", modelId,
                "runner", runnerName,
                "status", "success");

        // Increment success counter
        Counter.builder(REQUESTS_SUCCESS)
                .tags(tags)
                .description("Number of successful inference requests")
                .register(registry)
                .increment();

        // Increment total counter
        Counter.builder(REQUESTS_TOTAL)
                .tags(tags.and("result", "success"))
                .description("Total number of inference requests")
                .register(registry)
                .increment();

        // Record latency
        Timer.builder(LATENCY)
                .tags(tags)
                .description("Inference latency in milliseconds")
                .publishPercentiles(0.5, 0.95, 0.99)
                .publishPercentileHistogram()
                .register(registry)
                .record(Duration.ofMillis(latencyMs));

        // Decrement active requests
        decrementActiveRequests(tenantId, modelId);

        log.debug("Metrics recorded: tenant={}, model={}, runner={}, latencyMs={}",
                tenantId, modelId, runnerName, latencyMs);
    }

    /**
     * Record failed inference request.
     */
    public void recordFailure(String tenantId, String modelId, String errorType) {
        Tags tags = Tags.of(
                "tenant", tenantId,
                "model", modelId,
                "error_type", errorType,
                "status", "failure");

        // Increment failure counter
        Counter.builder(REQUESTS_FAILURE)
                .tags(tags)
                .description("Number of failed inference requests")
                .register(registry)
                .increment();

        // Increment total counter
        Counter.builder(REQUESTS_TOTAL)
                .tags(tags.and("result", "failure"))
                .description("Total number of inference requests")
                .register(registry)
                .increment();

        // Decrement active requests
        decrementActiveRequests(tenantId, modelId);
    }

    /**
     * Record request started (increment active requests).
     */
    public void recordRequestStarted(String tenantId, String modelId) {
        incrementActiveRequests(tenantId, modelId);
    }

    /**
     * Record input data size.
     */
    public void recordInputSize(String tenantId, String modelId, long bytes) {
        DistributionSummary.builder(INPUT_SIZE)
                .tags("tenant", tenantId, "model", modelId)
                .description("Size of input data in bytes")
                .baseUnit("bytes")
                .register(registry)
                .record(bytes);
    }

    /**
     * Record output data size.
     */
    public void recordOutputSize(String tenantId, String modelId, long bytes) {
        DistributionSummary.builder(OUTPUT_SIZE)
                .tags("tenant", tenantId, "model", modelId)
                .description("Size of output data in bytes")
                .baseUnit("bytes")
                .register(registry)
                .record(bytes);
    }

    /**
     * Record runner health status.
     */
    public void recordRunnerHealth(String runnerName, String deviceType, boolean isHealthy) {
        Gauge.builder(RUNNER_HEALTH, () -> isHealthy ? 1.0 : 0.0)
                .tags("runner", runnerName, "device", deviceType)
                .description("Runner health status (1=healthy, 0=unhealthy)")
                .register(registry);
    }

    /**
     * Record queue size.
     */
    public void recordQueueSize(String queueName, int size) {
        Gauge.builder(QUEUE_SIZE, () -> size)
                .tags("queue", queueName)
                .description("Number of items in queue")
                .register(registry);
    }

    /**
     * Record quota usage.
     */
    public void recordQuotaUsage(String tenantId, String resourceType, long used, long limit) {
        Tags tags = Tags.of("tenant", tenantId, "resource", resourceType);

        Gauge.builder(QUOTA_USAGE, () -> used)
                .tags(tags)
                .description("Current quota usage")
                .register(registry);

        Gauge.builder(QUOTA_LIMIT, () -> limit)
                .tags(tags)
                .description("Quota limit")
                .register(registry);
    }

    /**
     * Get current active request count.
     */
    public long getActiveRequestCount(String tenantId, String modelId) {
        String key = tenantId + ":" + modelId;
        AtomicLong counter = activeRequests.get(key);
        return counter != null ? counter.get() : 0;
    }

    /**
     * Create a custom timer for specific operations.
     */
    public Timer.Sample startTimer() {
        return Timer.start(registry);
    }

    /**
     * Stop timer and record with tags.
     */
    public void stopTimer(Timer.Sample sample, String operation, Tags tags) {
        sample.stop(Timer.builder(operation)
                .tags(tags)
                .register(registry));
    }

    // ===== Private Helper Methods =====

    private void incrementActiveRequests(String tenantId, String modelId) {
        String key = tenantId + ":" + modelId;
        AtomicLong counter = activeRequests.computeIfAbsent(key, k -> {
            AtomicLong newCounter = new AtomicLong(0);

            // Register gauge for this tenant+model combination
            Gauge.builder(ACTIVE_REQUESTS, newCounter, AtomicLong::get)
                    .tags("tenant", tenantId, "model", modelId)
                    .description("Number of active inference requests")
                    .register(registry);

            return newCounter;
        });

        counter.incrementAndGet();
    }

    private void decrementActiveRequests(String tenantId, String modelId) {
        String key = tenantId + ":" + modelId;
        AtomicLong counter = activeRequests.get(key);
        if (counter != null && counter.get() > 0) {
            counter.decrementAndGet();
        }
    }

    /**
     * Get metrics summary for monitoring dashboard.
     */
    public MetricsSummary getSummary(String tenantId, String modelId) {
        String key = tenantId + ":" + modelId;

        // These would ideally come from actual metric queries
        // For now, returning current active count
        return new MetricsSummary(
                tenantId,
                modelId,
                getActiveRequestCount(tenantId, modelId),
                0, // Total requests (would query from Counter)
                0, // Success count
                0, // Failure count
                0.0, // Average latency
                0.0, // P95 latency
                0.0 // P99 latency
        );
    }

    public record MetricsSummary(
            String tenantId,
            String modelId,
            long activeRequests,
            long totalRequests,
            long successCount,
            long failureCount,
            double avgLatencyMs,
            double p95LatencyMs,
            double p99LatencyMs) {
        public double successRate() {
            return totalRequests > 0
                    ? (successCount * 100.0) / totalRequests
                    : 0.0;
        }

        public double failureRate() {
            return totalRequests > 0
                    ? (failureCount * 100.0) / totalRequests
                    : 0.0;
        }
    }
}
