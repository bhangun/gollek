package tech.kayys.gollek.sdk.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Request for model inference.
 *
 * <p>Use {@link #builder()} or the convenience factory {@link #of(String)} to construct instances.
 *
 * <pre>{@code
 * GenerationRequest req = GenerationRequest.builder()
 *         .prompt("Explain quantum entanglement")
 *         .model("llama3")
 *         .maxTokens(256)
 *         .temperature(0.8f)
 *         .build();
 * }</pre>
 *
 * @param prompt      the input text to send to the model
 * @param model       the model identifier; may be {@code null} to use the client default
 * @param maxTokens   maximum number of tokens to generate (default: 512)
 * @param temperature sampling temperature controlling randomness (default: 0.7)
 * @param topP        nucleus sampling probability mass (default: 1.0)
 * @param stream      whether to stream tokens incrementally (default: false)
 * @param metadata    arbitrary key-value pairs forwarded to the provider
 * @param stop        list of stop sequences that terminate generation early
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonDeserialize(builder = GenerationRequest.Builder.class)
public record GenerationRequest(
        String prompt,
        String model,
        int maxTokens,
        float temperature,
        float topP,
        boolean stream,
        Map<String, Object> metadata,
        List<String> stop
) {

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a request with only a prompt, using all other defaults.
     *
     * @param prompt the input text
     * @return a minimal {@link GenerationRequest}
     */
    public static GenerationRequest of(String prompt) {
        return builder().prompt(prompt).build();
    }

    @JsonPOJOBuilder(withPrefix = "")
    public static class Builder {
        private String prompt;
        private String model;
        private int maxTokens = 512;
        private float temperature = 0.7f;
        private float topP = 1.0f;
        private boolean stream = false;
        private Map<String, Object> metadata = Collections.emptyMap();
        private List<String> stop = Collections.emptyList();

        public Builder prompt(String prompt) {
            this.prompt = prompt;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder maxTokens(int maxTokens) {
            this.maxTokens = maxTokens;
            return this;
        }

        public Builder temperature(float temperature) {
            this.temperature = temperature;
            return this;
        }

        public Builder topP(float topP) {
            this.topP = topP;
            return this;
        }

        public Builder stream(boolean stream) {
            this.stream = stream;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public Builder stop(List<String> stop) {
            this.stop = stop;
            return this;
        }

        public GenerationRequest build() {
            return new GenerationRequest(prompt, model, maxTokens, temperature, topP, stream, metadata, stop);
        }
    }
}
