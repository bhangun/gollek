package tech.kayys.golek.engine.inference;

import java.time.LocalDateTime;

public record EngineStats(
        String engineId,
        String modelId,
        long totalRequests,
        long activeRequests,
        double avgLatencyMs,
        double p95LatencyMs,
        double p99LatencyMs,
        long totalTokensProcessed,
        double tokensPerSecond,
        LocalDateTime lastActivity,
        String status
) {
}