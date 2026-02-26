package tech.kayys.gollek.spi.inference;

/**
 * Tracks performance and size metrics for batching operations.
 * <p>
 * Enriched fields enable scheduler-level observability across all three
 * batching strategies (static, dynamic, continuous).
 */
public record BatchMetrics(
                int totalRequests,
                int totalBatches,
                int currentQueueDepth,
                long currentQueueLatencyMs,
                double avgBatchSize,
                double throughputRps,
                long avgBatchLatencyMs) {

        /** Convenience constructor for callers that don't yet track the new fields. */
        public BatchMetrics(int totalRequests, int totalBatches, int currentQueueDepth, long currentQueueLatencyMs) {
                this(totalRequests, totalBatches, currentQueueDepth, currentQueueLatencyMs, 0.0, 0.0, 0L);
        }

        public static BatchMetrics empty() {
                return new BatchMetrics(0, 0, 0, 0L, 0.0, 0.0, 0L);
        }
}
