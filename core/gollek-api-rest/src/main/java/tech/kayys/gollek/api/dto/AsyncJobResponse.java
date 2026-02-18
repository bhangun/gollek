package tech.kayys.gollek.api.dto;

/**
 * Response model for async job submission.
 */
public record AsyncJobResponse(String jobId, String requestId) {
}
