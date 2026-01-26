package tech.kayys.golek.core.observability;

import io.micrometer.core.instrument.*;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Centralized metrics collection for inference operations.
 * Tracks request counts, durations, errors, and custom business metrics.
 */
@ApplicationScoped
public class InferenceMetricsCollector {
    
    private static final Logger LOG = Logger.getLogger(InferenceMetricsCollector.class);
    
    @Inject
    MeterRegistry meterRegistry;
    
    private Counter successCounter;
    private Counter failureCounter;
    private Timer requestTimer;
    private final Map<String, Counter> errorCounters = new ConcurrentHashMap<>();
    private final Map<String, Counter> retryCounters = new ConcurrentHashMap<>();
    
    @jakarta.annotation.PostConstruct
    void init() {
        LOG.info("Initializing InferenceMetricsCollector");
        
        successCounter = Counter.builder("inference.requests.success")
            .description("Successful inference requests")
            .tag("component", "inference-engine")
            .register(meterRegistry);
            
        failureCounter = Counter.builder("inference.requests.failure")
            .description("Failed inference requests")
            .tag("component", "inference-engine")
            .register(meterRegistry);
            
        requestTimer = Timer.builder("inference.request.duration")
            .description("Inference request duration")
            .tag("component", "inference-engine")
            .register(meterRegistry);
            
        LOG.info("InferenceMetricsCollector initialized");
    }
    
    /**
     * Record a successful inference request
     */
    public void recordSuccess(Duration duration) {
        successCounter.increment();
        requestTimer.record(duration);
        LOG.debugf("Recorded successful request: %d ms", duration.toMillis());
    }
    
    /**
     * Record a successful inference request with tags
     */
    public void recordSuccess(Duration duration, Map<String, String> tags) {
        successCounter.increment();
        requestTimer.record(duration);
        LOG.debugf("Recorded successful request with tags %s: %d ms", tags, duration.toMillis());
    }
    
    /**
     * Record a failed inference request
     */
    public void recordFailure(String errorType) {
        failureCounter.increment();
        
        errorCounters.computeIfAbsent(errorType, 
            type -> Counter.builder("inference.errors")
                .description("Inference errors by type")
                .tag("error_type", type)
                .tag("component", "inference-engine")
                .register(meterRegistry))
            .increment();
            
        LOG.debugf("Recorded failure: %s", errorType);
    }
    
    /**
     * Record a failed inference request with duration
     */
    public void recordFailure(String errorType, Duration duration) {
        recordFailure(errorType);
        requestTimer.record(duration);
    }
    
    /**
     * Record a retry attempt
     */
    public void recordRetry(int attemptNumber) {
        String attemptKey = "attempt_" + attemptNumber;
        retryCounters.computeIfAbsent(attemptKey,
            key -> Counter.builder("inference.retries")
                .description("Inference retry attempts")
                .tag("attempt", String.valueOf(attemptNumber))
                .tag("component", "inference-engine")
                .register(meterRegistry))
            .increment();
            
        LOG.debugf("Recorded retry attempt: %d", attemptNumber);
    }
    
    /**
     * Record pipeline phase execution
     */
    public void recordPhaseExecution(String phaseName, Duration duration, boolean success) {
        Timer phaseTimer = Timer.builder("inference.phase.duration")
            .description("Inference phase execution duration")
            .tag("phase", phaseName)
            .tag("success", String.valueOf(success))
            .tag("component", "inference-pipeline")
            .register(meterRegistry);
            
        phaseTimer.record(duration);
        LOG.debugf("Recorded phase %s execution: %d ms (success=%s)", 
                   phaseName, duration.toMillis(), success);
    }
    
    /**
     * Record plugin execution
     */
    public void recordPluginExecution(String pluginId, String phase, Duration duration, boolean success) {
        Timer pluginTimer = Timer.builder("inference.plugin.duration")
            .description("Plugin execution duration")
            .tag("plugin", pluginId)
            .tag("phase", phase)
            .tag("success", String.valueOf(success))
            .tag("component", "inference-pipeline")
            .register(meterRegistry);
            
        pluginTimer.record(duration);
        LOG.debugf("Recorded plugin %s execution in phase %s: %d ms (success=%s)", 
                   pluginId, phase, duration.toMillis(), success);
    }
    
    /**
     * Get current metrics summary
     */
    public MetricsSummary getSummary() {
        return new MetricsSummary(
            (long) successCounter.count(),
            (long) failureCounter.count(),
            requestTimer.count(),
            requestTimer.mean(java.util.concurrent.TimeUnit.MILLISECONDS)
        );
    }
    
    /**
     * Metrics summary record
     */
    public record MetricsSummary(
        long successCount,
        long failureCount,
        long totalRequests,
        double averageDurationMs
    ) {
        public double successRate() {
            return totalRequests > 0 ? (double) successCount / totalRequests : 0.0;
        }
    }
}
