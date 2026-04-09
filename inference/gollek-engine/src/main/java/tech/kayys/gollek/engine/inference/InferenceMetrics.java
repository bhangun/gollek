package tech.kayys.gollek.engine.inference;

import io.micrometer.core.instrument.*;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

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
 * <li>Throughput metrics (tokens per second, requests per second)</li>
 * <li>Streaming metrics (Time to First Token - TTFT)</li>
 * <li>Resource metrics (concurrent requests)</li>
 * </ul>
 *
 * <p>
 * All metrics are exported to Prometheus via Micrometer.
 */
@ApplicationScoped
public class InferenceMetrics {

    private static final Logger log = Logger.getLogger(InferenceMetrics.class);

    @Inject
    MeterRegistry registry;

    // Active request tracking
    private final ConcurrentHashMap<String, AtomicLong> activeRequests = new ConcurrentHashMap<>();

    // Metric names
    private static final String REQUESTS_TOTAL = "inference.requests.total";
    private static final String REQUESTS_SUCCESS = "inference.requests.success";
    private static final String REQUESTS_FAILURE = "inference.requests.failure";
    private static final String LATENCY = "inference.latency";
    private static final String TTFT = "inference.streaming.ttft";
    private static final String TOKENS_INPUT = "inference.tokens.input";
    private static final String TOKENS_OUTPUT = "inference.tokens.output";
    private static final String ACTIVE_REQUESTS = "inference.requests.active";
    private static final String TPOT = "inference.streaming.tpot";
    private static final String ITL = "inference.streaming.itl";
    
    // Additional enterprise tracking metrics
    private static final String INFERENCE_TOKENS_TOTAL = "gollek.inference.tokens.total";
    private static final String INFERENCE_ITL = "gollek.inference.itl";
    private static final String INFERENCE_REQUESTS = "gollek.inference.requests";
    
    private static final String INPUT_SIZE = "inference.input.bytes";
    private static final String OUTPUT_SIZE = "inference.output.bytes";
    private static final String RUNNER_HEALTH = "inference.runner.health";
    private static final String QUOTA_USAGE = "inference.quota.usage";
    private static final String QUOTA_LIMIT = "inference.quota.limit";

    /**
     * Record successful inference request with latency.
     */
    public void recordSuccess(String tenantId, String modelId, String runnerName, long latencyMs) {
        recordSuccess(tenantId, modelId, runnerName, latencyMs, 0, 0);
    }

    /**
     * Record successful inference request with latency and token usage.
     */
    public void recordSuccess(String tenantId, String modelId, String runnerName, long latencyMs, long inputTokens, long outputTokens) {
        Tags tags = Tags.of(
                "tenant", tenantId,
                "model", modelId,
                "runner", runnerName,
                "status", "success");

        // Increment success and total counters
        registry.counter(REQUESTS_SUCCESS, tags).increment();
        registry.counter(REQUESTS_TOTAL, tags.and("result", "success")).increment();

        // Record latency
        Timer.builder(LATENCY)
                .tags(tags)
                .description("Inference latency in milliseconds")
                .publishPercentiles(0.5, 0.95, 0.99)
                .register(registry)
                .record(Duration.ofMillis(latencyMs));

        // Record tokens if available
        if (inputTokens > 0) {
            registry.counter(TOKENS_INPUT, tags).increment(inputTokens);
        }
        if (outputTokens > 0) {
            registry.counter(TOKENS_OUTPUT, tags).increment(outputTokens);
        }

        decrementActiveRequests(tenantId, modelId);

        log.debugf("Metrics recorded: tenant=%s, model=%s, latency=%dms, tokens(in/out)=%d/%d",
                tenantId, modelId, latencyMs, inputTokens, outputTokens);
    }

    /**
     * Record Time to First Token (TTFT) for streaming requests.
     */
    public void recordTTFT(String tenantId, String modelId, String runnerName, long ttftMs) {
        Tags tags = Tags.of("tenant", tenantId, "model", modelId, "runner", runnerName);
        
        Timer.builder(TTFT)
                .tags(tags)
                .description("Time to first token in milliseconds")
                .publishPercentiles(0.5, 0.95, 0.99)
                .register(registry)
                .record(Duration.ofMillis(ttftMs));
    }

    /**
     * Record Inter-Token Latency (ITL) on the fly without array allocations.
     */
    public void recordInterTokenLatency(String tenantId, String modelId, String runnerName, long gapNanos) {
        if (gapNanos > 0) {
            Tags tags = Tags.of("tenant", tenantId, "model", modelId, "runner", runnerName);
            registry.timer(INFERENCE_ITL, tags).record(Duration.ofNanos(gapNanos));
        }
    }

    /**
     * Record Enterprise streaming metrics upon request completion.
     * Computes TPOT and Token Count.
     */
    public void recordStreamCompletionStats(String tenantId, String modelId, String runnerName, 
                                            int inputTokens, int outputTokens, 
                                            long totalDecodeNanos) {
                                        
        Tags tags = Tags.of("tenant", tenantId, "model", modelId, "runner", runnerName);
        
        // 1. Throughput (System-wide)
        registry.counter(INFERENCE_TOKENS_TOTAL, tags.and("type", "input")).increment(inputTokens);
        registry.counter(INFERENCE_TOKENS_TOTAL, tags.and("type", "output")).increment(outputTokens);
        
        // Success counter
        registry.counter(INFERENCE_REQUESTS, tags.and("status", "success")).increment();
        
        // TPOT (exclude the first token from time computation as it is TTFT bounded)
        if (outputTokens > 1 && totalDecodeNanos > 0) {
            long decodeDurationMs = java.util.concurrent.TimeUnit.NANOSECONDS.toMillis(totalDecodeNanos);
            double tpot = (double) decodeDurationMs / (outputTokens - 1);
            DistributionSummary.builder(TPOT)
                    .tags(tags)
                    .description("Time per output token")
                    .baseUnit("milliseconds")
                    .register(registry)
                    .record(tpot);
        }
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

        registry.counter(REQUESTS_FAILURE, tags).increment();
        registry.counter(REQUESTS_TOTAL, tags.and("result", "failure")).increment();

        decrementActiveRequests(tenantId, modelId);
    }

    /**
     * Record request started.
     */
    public void recordRequestStarted(String tenantId, String modelId) {
        incrementActiveRequests(tenantId, modelId);
    }

    /**
     * Record input data size in bytes.
     */
    public void recordInputSize(String tenantId, String modelId, long bytes) {
        DistributionSummary.builder(INPUT_SIZE)
                .tags("tenant", tenantId, "model", modelId)
                .baseUnit("bytes")
                .register(registry)
                .record(bytes);
    }

    /**
     * Record output data size in bytes.
     */
    public void recordOutputSize(String tenantId, String modelId, long bytes) {
        DistributionSummary.builder(OUTPUT_SIZE)
                .tags("tenant", tenantId, "model", modelId)
                .baseUnit("bytes")
                .register(registry)
                .record(bytes);
    }

    /**
     * Record runner health status.
     */
    public void recordRunnerHealth(String runnerName, String deviceType, boolean isHealthy) {
        registry.gauge(RUNNER_HEALTH, Tags.of("runner", runnerName, "device", deviceType), isHealthy ? 1.0 : 0.0);
    }

    /**
     * Record queue size.
     */
    public void recordQueueSize(String queueName, int size) {
        registry.gauge("inference.queue.size", Tags.of("queue", queueName), size);
    }

    /**
     * Record quota usage.
     */
    public void recordQuotaUsage(String tenantId, String resourceType, long used, long limit) {
        Tags tags = Tags.of("tenant", tenantId, "resource", resourceType);
        registry.gauge(QUOTA_USAGE, tags, used);
        registry.gauge(QUOTA_LIMIT, tags, limit);
    }

    private void incrementActiveRequests(String tenantId, String modelId) {
        String key = tenantId + ":" + modelId;
        activeRequests.computeIfAbsent(key, k -> {
            AtomicLong counter = new AtomicLong(0);
            Gauge.builder(ACTIVE_REQUESTS, counter, AtomicLong::get)
                    .tags("tenant", tenantId, "model", modelId)
                    .register(registry);
            return counter;
        }).incrementAndGet();
    }

    private void decrementActiveRequests(String tenantId, String modelId) {
        String key = tenantId + ":" + modelId;
        AtomicLong counter = activeRequests.get(key);
        if (counter != null && counter.get() > 0) {
            counter.decrementAndGet();
        }
    }

    /**
     * Get real-time metrics summary for a specific tenant and model.
     */
    public MetricsSummary getSummary(String tenantId, String modelId) {
        Tags tags = Tags.of("tenant", tenantId, "model", modelId);
        
        double success = registry.find(REQUESTS_SUCCESS).tags(tags).counter() != null ? 
                        registry.find(REQUESTS_SUCCESS).tags(tags).counter().count() : 0;
        double failure = registry.find(REQUESTS_FAILURE).tags(tags).counter() != null ? 
                        registry.find(REQUESTS_FAILURE).tags(tags).counter().count() : 0;
        
        Timer timer = registry.find(LATENCY).tags(tags).timer();
        double avgLatency = timer != null ? timer.mean(java.util.concurrent.TimeUnit.MILLISECONDS) : 0;
        
        return new MetricsSummary(
                tenantId,
                modelId,
                activeRequests.getOrDefault(tenantId + ":" + modelId, new AtomicLong(0)).get(),
                (long)(success + failure),
                (long)success,
                (long)failure,
                avgLatency,
                0, 0 // P95/P99 would require more complex registry querying
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
            return totalRequests > 0 ? (successCount * 100.0) / totalRequests : 0.0;
        }
    }
}

