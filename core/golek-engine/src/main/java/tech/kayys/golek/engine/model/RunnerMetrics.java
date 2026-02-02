package tech.kayys.golek.engine.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Runtime metrics for a model runner.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
class RunnerMetrics {

    /**
     * Total requests processed.
     */
    private long totalRequests;

    /**
     * Failed requests count.
     */
    private long failedRequests;

    /**
     * Average latency in milliseconds.
     */
    private long averageLatencyMs;

    /**
     * P95 latency in milliseconds.
     */
    private long p95LatencyMs;

    /**
     * P99 latency in milliseconds.
     */
    private long p99LatencyMs;
}
