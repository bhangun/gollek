package tech.kayys.gollek.spi.inference;

import java.util.List;

/**
 * Represents the response from a batched inference request.
 */
public record BatchResponse(
        String batchId,
        List<BatchResult> results,
        BatchMetrics metrics) {

    public int successCount() {
        if (results == null)
            return 0;
        return (int) results.stream().filter(BatchResult::succeeded).count();
    }

    public int failureCount() {
        if (results == null)
            return 0;
        return (int) results.stream().filter(r -> !r.succeeded()).count();
    }
}
