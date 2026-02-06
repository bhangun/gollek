package tech.kayys.golek.spi.model;

/**
 * Resource utilization metrics for a model runner
 */
public record ResourceMetrics(
        long cpuUsagePercent,
        long memoryUsageBytes,
        long gpuUsagePercent,
        long vramUsageBytes,
        int activeRequests) {
}
