package tech.kayys.gollek.sdk.core.model;

import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.time.Instant;

/**
 * Represents the status of an async inference job.
 */
public class AsyncJobStatus {
    private final String jobId;
    private final String requestId;
    private final String status; // PENDING, RUNNING, COMPLETED, FAILED
    private final InferenceResponse result;
    private final String error;
    private final Instant submittedAt;
    private final Instant completedAt;

    public AsyncJobStatus(String jobId, String requestId, String status,
            InferenceResponse result, String error, Instant submittedAt, Instant completedAt) {
        this.jobId = jobId;
        this.requestId = requestId;
        this.status = status;
        this.result = result;
        this.error = error;
        this.submittedAt = submittedAt;
        this.completedAt = completedAt;
    }

    public String getJobId() {
        return jobId;
    }

    public String getApiKey() {
        if (requestId == null || requestId.isBlank()) {
            return "community";
        }
        return requestId;
    }

    /**
     * @deprecated Tenant ID is resolved server-side from the API key.
     *             Client code should not rely on this field.
     */
    @Deprecated
    public String getRequestId() {
        return requestId;
    }

    public String getStatus() {
        return status;
    }

    public InferenceResponse getResult() {
        return result;
    }

    public String getError() {
        return error;
    }

    public Instant getSubmittedAt() {
        return submittedAt;
    }

    public Instant getCompletedAt() {
        return completedAt;
    }

    public boolean isComplete() {
        return "COMPLETED".equals(status) || "FAILED".equals(status);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String jobId;
        private String requestId;
        private String status;
        private InferenceResponse result;
        private String error;
        private Instant submittedAt;
        private Instant completedAt;

        public Builder jobId(String jobId) {
            this.jobId = jobId;
            return this;
        }

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder apiKey(String apiKey) {
            this.requestId = apiKey;
            return this;
        }

        public Builder status(String status) {
            this.status = status;
            return this;
        }

        public Builder result(InferenceResponse result) {
            this.result = result;
            return this;
        }

        public Builder error(String error) {
            this.error = error;
            return this;
        }

        public Builder submittedAt(Instant submittedAt) {
            this.submittedAt = submittedAt;
            return this;
        }

        public Builder completedAt(Instant completedAt) {
            this.completedAt = completedAt;
            return this;
        }

        public AsyncJobStatus build() {
            return new AsyncJobStatus(jobId, requestId, status, result, error, submittedAt, completedAt);
        }
    }
}
