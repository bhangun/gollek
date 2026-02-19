package tech.kayys.gollek.spi.inference;

/**
 * Tracks performance and size metrics for batching operations.
 */
public record BatchMetrics(
        int totalRequests,
        int totalBatches,
        int currentQueueDepth,
        long currentQueueLatencyMs) {
}
