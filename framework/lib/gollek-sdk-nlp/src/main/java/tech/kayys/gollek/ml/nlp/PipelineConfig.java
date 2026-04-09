package tech.kayys.gollek.ml.nlp;

/**
 * Immutable configuration for NLP pipeline creation.
 *
 * <p>Use {@link #builder(String, String)} to construct instances. All fields have
 * sensible defaults so only task and model are required.
 *
 * @param task        pipeline task identifier (e.g. {@code "text-generation"})
 * @param modelId     model identifier to load (e.g. {@code "Qwen/Qwen2.5-0.5B"})
 * @param maxLength   maximum number of tokens to generate (default: 512)
 * @param temperature sampling temperature controlling output randomness (default: 0.7)
 * @param topP        nucleus sampling cumulative probability threshold (default: 0.9)
 * @param topK        top-k sampling candidate count (default: 50)
 * @param stream      whether to stream tokens incrementally (default: false)
 * @param device      target compute device, e.g. {@code "cpu"} or {@code "cuda"} (default: {@code "cpu"})
 */
public record PipelineConfig(
    String task,
    String modelId,
    int maxLength,
    float temperature,
    float topP,
    int topK,
    boolean stream,
    String device
) {
    /**
     * Creates a new builder for the given task and model.
     *
     * @param task    pipeline task identifier
     * @param modelId model identifier
     * @return a new {@link Builder}
     */
    public static Builder builder(String task, String modelId) {
        return new Builder(task, modelId);
    }

    /**
     * Builder for {@link PipelineConfig}.
     */
    public static class Builder {
        private final String task;
        private final String modelId;
        private int maxLength = 512;
        private float temperature = 0.7f;
        private float topP = 0.9f;
        private int topK = 50;
        private boolean stream = false;
        private String device = "cpu";

        Builder(String task, String modelId) {
            this.task = task;
            this.modelId = modelId;
        }

        /** @param v maximum tokens to generate; must be &gt; 0 */
        public Builder maxLength(int v) { this.maxLength = v; return this; }

        /** @param v sampling temperature in range {@code [0.0, 2.0]} */
        public Builder temperature(float v) { this.temperature = v; return this; }

        /** @param v nucleus sampling probability in range {@code (0.0, 1.0]} */
        public Builder topP(float v) { this.topP = v; return this; }

        /** @param v number of top candidates to consider during sampling */
        public Builder topK(int v) { this.topK = v; return this; }

        /** @param v {@code true} to enable incremental token streaming */
        public Builder stream(boolean v) { this.stream = v; return this; }

        /** @param v target device, e.g. {@code "cpu"}, {@code "cuda"}, {@code "metal"} */
        public Builder device(String v) { this.device = v; return this; }

        /**
         * Builds the {@link PipelineConfig}.
         *
         * @return a new immutable {@link PipelineConfig}
         */
        public PipelineConfig build() {
            return new PipelineConfig(task, modelId, maxLength, temperature, topP, topK, stream, device);
        }
    }
}
