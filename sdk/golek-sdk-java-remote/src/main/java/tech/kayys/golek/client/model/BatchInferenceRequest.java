package tech.kayys.golek.client.model;

import tech.kayys.golek.api.inference.InferenceRequest;

import java.util.List;

/**
 * Wrapper for batch inference requests.
 */
public class BatchInferenceRequest {
    private final List<InferenceRequest> requests;
    private final Integer maxConcurrent;

    public BatchInferenceRequest(List<InferenceRequest> requests, Integer maxConcurrent) {
        this.requests = requests;
        this.maxConcurrent = maxConcurrent;
    }

    public List<InferenceRequest> getRequests() {
        return requests;
    }

    public Integer getMaxConcurrent() {
        return maxConcurrent;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private List<InferenceRequest> requests;
        private Integer maxConcurrent;

        public Builder requests(List<InferenceRequest> requests) {
            this.requests = requests;
            return this;
        }

        public Builder maxConcurrent(Integer maxConcurrent) {
            this.maxConcurrent = maxConcurrent;
            return this;
        }

        public BatchInferenceRequest build() {
            return new BatchInferenceRequest(requests, maxConcurrent);
        }
    }
}