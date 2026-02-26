package tech.kayys.gollek.spi.inference;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;

import tech.kayys.gollek.spi.auth.ApiKeyConstants;

/**
 * Detailed status of an asynchronous inference job.
 */
public record AsyncJobStatus(
        String jobId,
        String requestId,
        @Deprecated String apiKey,
        String status,
        InferenceResponse result,
        String error,
        Instant submittedAt,
        Instant completedAt) {

    @JsonCreator
    public AsyncJobStatus(
            @JsonProperty("jobId") String jobId,
            @JsonProperty("requestId") String requestId,
            @JsonProperty("apiKey") @Deprecated String apiKey,
            @JsonProperty("status") String status,
            @JsonProperty("result") InferenceResponse result,
            @JsonProperty("error") String error,
            @JsonProperty("submittedAt") Instant submittedAt,
            @JsonProperty("completedAt") Instant completedAt) {
        this.jobId = jobId;
        this.requestId = requestId;
        this.apiKey = apiKey;
        this.status = status;
        this.result = result;
        this.error = error;
        this.submittedAt = submittedAt;
        this.completedAt = completedAt;
    }

    public AsyncJobStatus(String jobId, String requestId, String status,
            InferenceResponse result, String error, Instant submittedAt, Instant completedAt) {
        this(jobId, requestId, null, status, result, error, submittedAt, completedAt);
    }

    /**
     * Check if the job has finished processing (success, failure, or cancelled).
     */
    public boolean isComplete() {
        return "COMPLETED".equals(status) || "FAILED".equals(status) || "CANCELLED".equals(status);
    }

    public String apiKey() {
        if (apiKey == null || apiKey.isBlank()) {
            return ApiKeyConstants.COMMUNITY_API_KEY;
        }
        return apiKey;
    }
}
