package tech.kayys.golek.spi.inference;

import java.time.Instant;

/**
 * Detailed status of an asynchronous inference job.
 */
public record AsyncJobStatus(
        String jobId,
        String requestId,
        String tenantId,
        String status,
        InferenceResponse result,
        String error,
        Instant submittedAt,
        Instant completedAt) {

    /**
     * Check if the job has finished processing (success, failure, or cancelled).
     */
    public boolean isComplete() {
        return "COMPLETED".equals(status) || "FAILED".equals(status) || "CANCELLED".equals(status);
    }
}