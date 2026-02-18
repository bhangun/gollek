package tech.kayys.gollek.engine.observability;

import jakarta.enterprise.context.ApplicationScoped;

import java.time.Duration;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

@ApplicationScoped
public class RuntimeMetricsCache {

    private final Map<String, ProviderMetrics> providerMetrics = new ConcurrentHashMap<>();

    public void recordSuccess(String providerId, String modelId, long durationMs) {
        String key = providerId + ":" + modelId;
        providerMetrics.computeIfAbsent(key, k -> new ProviderMetrics())
                .recordSuccess(durationMs);
    }

    public void recordFailure(String providerId, String modelId, String errorType) {
        String key = providerId + ":" + modelId;
        providerMetrics.computeIfAbsent(key, k -> new ProviderMetrics())
                .recordFailure(errorType);
    }

    public Optional<Duration> getP95Latency(String providerId, String modelId) {
        String key = providerId + ":" + modelId;
        return Optional.ofNullable(providerMetrics.get(key))
                .flatMap(metrics -> metrics.getP95Latency());
    }

    public double getErrorRate(String providerId, Duration window) {
        // Search for all model metrics for this provider
        long total = 0;
        long successful = 0;

        for (Map.Entry<String, ProviderMetrics> entry : providerMetrics.entrySet()) {
            if (entry.getKey().startsWith(providerId + ":")) {
                total += entry.getValue().totalRequests;
                successful += entry.getValue().successfulRequests;
            }
        }

        if (total == 0)
            return 0.0;
        return (double) (total - successful) / total;
    }

    public double getCurrentLoad(String providerId) {
        // In a real implementation, this would track active requests
        return 0.2; // Low load placeholder
    }

    public boolean isCircuitBreakerOpen(String providerId) {
        return getErrorRate(providerId, Duration.ofMinutes(5)) > 0.5;
    }

    public static class ProviderMetrics {
        private volatile long totalRequests = 0;
        private volatile long successfulRequests = 0;
        private volatile long totalLatency = 0;
        private volatile long p95Latency = 0;

        public void recordSuccess(long durationMs) {
            totalRequests++;
            successfulRequests++;
            totalLatency += durationMs;

            // Simple approximation for P95 latency
            p95Latency = (long) (p95Latency * 0.9 + durationMs * 0.1);
        }

        public void recordFailure(String errorType) {
            totalRequests++;
        }

        public Optional<Duration> getP95Latency() {
            if (p95Latency > 0) {
                return Optional.of(Duration.ofMillis(p95Latency));
            }
            return Optional.empty();
        }
    }
}