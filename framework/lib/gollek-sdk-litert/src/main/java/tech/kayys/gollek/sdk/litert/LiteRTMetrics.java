package tech.kayys.gollek.sdk.litert;

import lombok.Builder;
import lombok.Data;

/**
 * LiteRT performance metrics.
 */
@Data
@Builder
public class LiteRTMetrics {

    /**
     * Total number of inferences.
     */
    @Builder.Default
    private long totalInferences = 0;

    /**
     * Number of failed inferences.
     */
    @Builder.Default
    private long failedInferences = 0;

    /**
     * Average latency in milliseconds.
     */
    @Builder.Default
    private double avgLatencyMs = 0.0;

    /**
     * P50 latency in milliseconds.
     */
    @Builder.Default
    private double p50LatencyMs = 0.0;

    /**
     * P95 latency in milliseconds.
     */
    @Builder.Default
    private double p95LatencyMs = 0.0;

    /**
     * P99 latency in milliseconds.
     */
    @Builder.Default
    private double p99LatencyMs = 0.0;

    /**
     * Peak memory usage in bytes.
     */
    @Builder.Default
    private long peakMemoryBytes = 0;

    /**
     * Current memory usage in bytes.
     */
    @Builder.Default
    private long currentMemoryBytes = 0;

    /**
     * Active delegate being used.
     */
    private String activeDelegate;
}
