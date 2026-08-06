package tech.kayys.alkhawarizm.spi.model;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Multimodal request for inference.
 * Supports multiple input modalities and output configurations.
 */
public class MultimodalRequest {

    private String requestId;
    private String model;
    private MultimodalContent[] inputs;
    private Map<String, Object> parameters;
    private OutputConfig outputConfig;
    private String tenantId;
    private long timeoutMs;

    public MultimodalRequest() {
    }

    public Set<ModalityType> inputModalities() {
        if (inputs == null) return Collections.emptySet();
        return Arrays.stream(inputs).map(MultimodalContent::getModality).collect(Collectors.toSet());
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

    public MultimodalContent[] getInputs() {
        return inputs;
    }

    public void setInputs(MultimodalContent[] inputs) {
        this.inputs = inputs;
    }

    public Map<String, Object> getParameters() {
        return parameters;
    }

    public void setParameters(Map<String, Object> parameters) {
        this.parameters = parameters;
    }

    public OutputConfig getOutputConfig() {
        return outputConfig;
    }

    public void setOutputConfig(OutputConfig outputConfig) {
        this.outputConfig = outputConfig;
    }

    public String getTenantId() {
        return tenantId;
    }

    public void setTenantId(String tenantId) {
        this.tenantId = tenantId;
    }

    public long getTimeoutMs() {
        return timeoutMs;
    }

    public void setTimeoutMs(long timeoutMs) {
        this.timeoutMs = timeoutMs;
    }

    /**
     * Output configuration for multimodal responses.
     */
    public static class OutputConfig {
        private ModalityType[] outputModalities;
        private int maxTokens;
        private double temperature;
        private double topP;
        private boolean stream;

        public OutputConfig() {
            this.outputModalities = new ModalityType[]{ModalityType.TEXT};
            this.maxTokens = 2048;
            this.temperature = 0.7;
            this.topP = 0.9;
            this.stream = false;
        }

        public ModalityType[] getOutputModalities() {
            return outputModalities;
        }

        public void setOutputModalities(ModalityType[] outputModalities) {
            this.outputModalities = outputModalities;
        }

        public int getMaxTokens() {
            return maxTokens;
        }

        public void setMaxTokens(int maxTokens) {
            this.maxTokens = maxTokens;
        }

        public double getTemperature() {
            return temperature;
        }

        public void setTemperature(double temperature) {
            this.temperature = temperature;
        }

        public double getTopP() {
            return topP;
        }

        public void setTopP(double topP) {
            this.topP = topP;
        }

        public boolean isStream() {
            return stream;
        }

        public void setStream(boolean stream) {
            this.stream = stream;
        }

        public static Builder builder() {
            return new Builder();
        }

        public static class Builder {
            private ModalityType[] outputModalities = new ModalityType[]{ModalityType.TEXT};
            private int maxTokens = 2048;
            private double temperature = 0.7;
            private double topP = 0.9;
            private boolean stream = false;

            public Builder outputModalities(ModalityType... modalities) {
                this.outputModalities = modalities;
                return this;
            }

            public Builder maxTokens(int maxTokens) {
                this.maxTokens = maxTokens;
                return this;
            }

            public Builder temperature(double temperature) {
                this.temperature = temperature;
                return this;
            }

            public Builder topP(double topP) {
                this.topP = topP;
                return this;
            }

            public Builder stream(boolean stream) {
                this.stream = stream;
                return this;
            }

            public OutputConfig build() {
                OutputConfig config = new OutputConfig();
                config.outputModalities = this.outputModalities;
                config.maxTokens = this.maxTokens;
                config.temperature = this.temperature;
                config.topP = this.topP;
                config.stream = this.stream;
                return config;
            }
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String requestId;
        private String model;
        private MultimodalContent[] inputs;
        private Map<String, Object> parameters;
        private OutputConfig outputConfig;
        private String tenantId;
        private long timeoutMs = 30000;

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder inputs(MultimodalContent... inputs) {
            this.inputs = inputs;
            return this;
        }

        public Builder parameters(Map<String, Object> parameters) {
            this.parameters = parameters;
            return this;
        }

        public Builder outputConfig(OutputConfig outputConfig) {
            this.outputConfig = outputConfig;
            return this;
        }

        public Builder tenantId(String tenantId) {
            this.tenantId = tenantId;
            return this;
        }

        public Builder timeoutMs(long timeoutMs) {
            this.timeoutMs = timeoutMs;
            return this;
        }

        public MultimodalRequest build() {
            MultimodalRequest request = new MultimodalRequest();
            request.requestId = this.requestId;
            request.model = this.model;
            request.inputs = this.inputs;
            request.parameters = this.parameters;
            request.outputConfig = this.outputConfig;
            request.tenantId = this.tenantId;
            request.timeoutMs = this.timeoutMs;
            return request;
        }
    }
}
