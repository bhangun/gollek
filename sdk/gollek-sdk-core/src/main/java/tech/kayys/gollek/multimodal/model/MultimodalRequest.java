package tech.kayys.gollek.multimodal.model;

import java.util.Map;

/**
 * Represents a multimodal inference request containing one or more
 * {@link MultimodalContent} inputs and an {@link OutputConfig}.
 *
 * <p>Use {@link #builder()} to construct instances:
 * <pre>{@code
 * MultimodalRequest request = MultimodalRequest.builder()
 *     .model("qwen-vl")
 *     .inputs(MultimodalContent.ofText("What is in this image?"),
 *             MultimodalContent.ofBase64Image(imageBytes, "image/jpeg"))
 *     .outputConfig(MultimodalRequest.OutputConfig.builder()
 *         .maxTokens(512)
 *         .temperature(0.7)
 *         .build())
 *     .build();
 * }</pre>
 *
 * @see MultimodalContent
 * @see MultimodalResponse
 */
public class MultimodalRequest {

    private String requestId;
    private String model;
    private MultimodalContent[] inputs = new MultimodalContent[0];
    private Map<String, Object> parameters;
    private OutputConfig outputConfig;
    private String tenantId;
    private long timeoutMs;

    /**
     * Returns the unique request identifier.
     *
     * @return request ID, or {@code null} if not set
     */
    public String getRequestId() { return requestId; }

    /**
     * Returns the model identifier to use for inference.
     *
     * @return model ID
     */
    public String getModel() { return model; }

    /**
     * Returns the array of multimodal input content items.
     *
     * @return non-null array of inputs (may be empty)
     */
    public MultimodalContent[] getInputs() { return inputs; }

    /**
     * Returns provider-specific inference parameters.
     *
     * @return parameter map, or {@code null}
     */
    public Map<String, Object> getParameters() { return parameters; }

    /**
     * Returns the output configuration controlling generation behaviour.
     *
     * @return output config, or {@code null} for defaults
     */
    public OutputConfig getOutputConfig() { return outputConfig; }

    /**
     * Returns the tenant identifier for multi-tenant deployments.
     *
     * @return tenant ID, or {@code null}
     */
    public String getTenantId() { return tenantId; }

    /**
     * Returns the request timeout in milliseconds. {@code 0} means no explicit timeout.
     *
     * @return timeout in milliseconds
     */
    public long getTimeoutMs() { return timeoutMs; }

    /**
     * Creates a new builder for {@link MultimodalRequest}.
     *
     * @return a new {@link Builder}
     */
    public static Builder builder() { return new Builder(); }

    /**
     * Builder for {@link MultimodalRequest}.
     */
    public static class Builder {
        private final MultimodalRequest request = new MultimodalRequest();

        /** @param requestId unique identifier for this request */
        public Builder requestId(String requestId) { request.requestId = requestId; return this; }

        /** @param model model identifier to use for inference */
        public Builder model(String model) { request.model = model; return this; }

        /** @param inputs one or more multimodal content items; {@code null} is treated as empty */
        public Builder inputs(MultimodalContent... inputs) {
            request.inputs = inputs == null ? new MultimodalContent[0] : inputs;
            return this;
        }

        /** @param parameters provider-specific key-value inference parameters */
        public Builder parameters(Map<String, Object> parameters) { request.parameters = parameters; return this; }

        /** @param outputConfig generation output configuration */
        public Builder outputConfig(OutputConfig outputConfig) { request.outputConfig = outputConfig; return this; }

        /** @param tenantId tenant identifier for multi-tenant deployments */
        public Builder tenantId(String tenantId) { request.tenantId = tenantId; return this; }

        /** @param timeoutMs request timeout in milliseconds; {@code 0} for no timeout */
        public Builder timeoutMs(long timeoutMs) { request.timeoutMs = timeoutMs; return this; }

        /**
         * Builds the {@link MultimodalRequest}.
         *
         * @return a configured {@link MultimodalRequest}
         */
        public MultimodalRequest build() { return request; }
    }

    /**
     * Controls the generation output: modalities, token limits, and sampling parameters.
     */
    public static class OutputConfig {
        private ModalityType[] outputModalities = new ModalityType[] { ModalityType.TEXT };
        private int maxTokens = 2048;
        private double temperature = 0.7;
        private double topP = 0.9;
        private boolean stream;

        /**
         * Returns the requested output modalities (default: {@code [TEXT]}).
         *
         * @return array of output {@link ModalityType}s
         */
        public ModalityType[] getOutputModalities() { return outputModalities; }

        /**
         * Returns the maximum number of tokens to generate (default: 2048).
         *
         * @return max token count
         */
        public int getMaxTokens() { return maxTokens; }

        /**
         * Returns the sampling temperature (default: 0.7).
         *
         * @return temperature in range {@code [0.0, 2.0]}
         */
        public double getTemperature() { return temperature; }

        /**
         * Returns the nucleus sampling probability threshold (default: 0.9).
         *
         * @return top-p value in range {@code (0.0, 1.0]}
         */
        public double getTopP() { return topP; }

        /**
         * Returns whether incremental token streaming is enabled.
         *
         * @return {@code true} if streaming is requested
         */
        public boolean isStream() { return stream; }

        /**
         * Creates a new builder for {@link OutputConfig}.
         *
         * @return a new {@link Builder}
         */
        public static Builder builder() { return new Builder(); }

        /**
         * Builder for {@link OutputConfig}.
         */
        public static class Builder {
            private final OutputConfig config = new OutputConfig();

            /** @param modalities desired output modalities; {@code null} defaults to {@code [TEXT]} */
            public Builder outputModalities(ModalityType... modalities) {
                config.outputModalities = modalities == null ? new ModalityType[] { ModalityType.TEXT } : modalities;
                return this;
            }

            /** @param maxTokens maximum tokens to generate; must be &gt; 0 */
            public Builder maxTokens(int maxTokens) { config.maxTokens = maxTokens; return this; }

            /** @param temperature sampling temperature in range {@code [0.0, 2.0]} */
            public Builder temperature(double temperature) { config.temperature = temperature; return this; }

            /** @param topP nucleus sampling threshold in range {@code (0.0, 1.0]} */
            public Builder topP(double topP) { config.topP = topP; return this; }

            /** @param stream {@code true} to enable incremental token streaming */
            public Builder stream(boolean stream) { config.stream = stream; return this; }

            /**
             * Builds the {@link OutputConfig}.
             *
             * @return a configured {@link OutputConfig}
             */
            public OutputConfig build() { return config; }
        }
    }
}
