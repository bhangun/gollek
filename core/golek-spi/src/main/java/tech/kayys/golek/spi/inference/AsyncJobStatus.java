package tech.kayys.golek.spi.inference;

import java.time.Instant;

import tech.kayys.golek.spi.auth.ApiKeyConstants;

/**
 * Detailed status of an asynchronous inference job.
 */
public record AsyncJobStatus(
        String jobId,
        String requestId,
        @Deprecated String tenantId,
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

    public String apiKey() {
        if (tenantId == null || tenantId.isBlank()) {
            return ApiKeyConstants.COMMUNITY_API_KEY;
        }
        return tenantId;
    }
}
