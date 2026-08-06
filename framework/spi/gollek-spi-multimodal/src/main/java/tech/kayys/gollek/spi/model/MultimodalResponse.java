package tech.kayys.alkhawarizm.spi.model;

import java.util.HashMap;
import java.util.Map;

/**
 * Multimodal response from inference.
 */
public class MultimodalResponse {

    private String requestId;
    private String model;
    private MultimodalContent[] outputs;
    private Usage usage;
    private Map<String, Object> metadata;
    private long durationMs;
    private ResponseStatus status;

    public MultimodalResponse() {
        this.metadata = new HashMap<>();
        this.status = ResponseStatus.SUCCESS;
    }

    public String getRequestId() {
        return requestId;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public MultimodalContent[] getOutputs() {
        return outputs;
    }

    public void setOutputs(MultimodalContent[] outputs) {
        this.outputs = outputs;
    }

    public Usage getUsage() {
        return usage;
    }

    public void setUsage(Usage usage) {
        this.usage = usage;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }

    public long getDurationMs() {
        return durationMs;
    }

    public void setDurationMs(long durationMs) {
        this.durationMs = durationMs;
    }

    public ResponseStatus getStatus() {
        return status;
    }

    public void setStatus(ResponseStatus status) {
        this.status = status;
    }

    /**
     * Token usage information.
     */
    public static class Usage {
        private int inputTokens;
        private int outputTokens;
        private int totalTokens;

        public Usage() {
        }

        public Usage(int inputTokens, int outputTokens) {
            this.inputTokens = inputTokens;
            this.outputTokens = outputTokens;
            this.totalTokens = inputTokens + outputTokens;
        }

        public int getInputTokens() {
            return inputTokens;
        }

        public void setInputTokens(int inputTokens) {
            this.inputTokens = inputTokens;
        }

        public int getOutputTokens() {
            return outputTokens;
        }

        public void setOutputTokens(int outputTokens) {
            this.outputTokens = outputTokens;
        }

        public int getTotalTokens() {
            return totalTokens;
        }

        public void setTotalTokens(int totalTokens) {
            this.totalTokens = totalTokens;
        }
    }

    /**
     * Response status.
     */
    public enum ResponseStatus {
        SUCCESS,
        IN_PROGRESS,
        FALLBACK,
        ERROR,
        TIMEOUT,
        RATE_LIMITED
    }

    /**
     * Create an error response.
     */
    public static MultimodalResponse error(String requestId, String model, String errorCode, String errorMessage) {
        MultimodalResponse response = new MultimodalResponse();
        response.setRequestId(requestId);
        response.setModel(model);
        response.setStatus(ResponseStatus.ERROR);
        response.getMetadata().put("error_code", errorCode);
        response.getMetadata().put("error_message", errorMessage);
        return response;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String requestId;
        private String model;
        private MultimodalContent[] outputs;
        private Usage usage;
        private Map<String, Object> metadata = new HashMap<>();
        private long durationMs;
        private ResponseStatus status = ResponseStatus.SUCCESS;

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder from(MultimodalResponse original) {
            if (original != null) {
                this.requestId = original.getRequestId();
                this.model = original.getModel();
                this.outputs = original.getOutputs();
                this.usage = original.getUsage();
                if (original.getMetadata() != null) {
                    this.metadata.putAll(original.getMetadata());
                }
                this.durationMs = original.getDurationMs();
                this.status = original.getStatus();
            }
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder outputs(MultimodalContent... outputs) {
            this.outputs = outputs;
            return this;
        }

        public Builder usage(Usage usage) {
            this.usage = usage;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public Builder durationMs(long durationMs) {
            this.durationMs = durationMs;
            return this;
        }

        public Builder status(ResponseStatus status) {
            this.status = status;
            return this;
        }

        public MultimodalResponse build() {
            MultimodalResponse response = new MultimodalResponse();
            response.requestId = this.requestId;
            response.model = this.model;
            response.outputs = this.outputs;
            response.usage = this.usage;
            response.metadata = this.metadata;
            response.durationMs = this.durationMs;
            response.status = this.status;
            return response;
        }
    }
}
