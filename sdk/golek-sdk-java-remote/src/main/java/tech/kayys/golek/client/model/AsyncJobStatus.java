package tech.kayys.golek.client.model;

import tech.kayys.golek.spi.inference.InferenceResponse;

import java.time.Instant;

/**
 * Represents the status of an async inference job.
 */
public class AsyncJobStatus {
    private final String jobId;
    private final String requestId;
    private final String tenantId;
    private final String status; // PENDING, RUNNING, COMPLETED, FAILED
    private final InferenceResponse result;
    private final String error;
    private final Instant submittedAt;
    private final Instant completedAt;

    public AsyncJobStatus(String jobId, String requestId, String tenantId, String status, 
                         InferenceResponse result, String error, Instant submittedAt, Instant completedAt) {
        this.jobId = jobId;
        this.requestId = requestId;
        this.tenantId = tenantId;
        this.status = status;
        this.result = result;
        this.error = error;
        this.submittedAt = submittedAt;
        this.completedAt = completedAt;
    }

    public String getJobId() {
        return jobId;
    }

    public String getRequestId() {
        return requestId;
    }

    public String getTenantId() {
        return tenantId;
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
        private String tenantId;
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

        public Builder tenantId(String tenantId) {
            this.tenantId = tenantId;
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
            return new AsyncJobStatus(jobId, requestId, tenantId, status, result, error, submittedAt, completedAt);
        }
    }
}