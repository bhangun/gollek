package tech.kayys.gollek.spi.inference;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

/**
 * Request for generating embeddings from text inputs.
 */
public record EmbeddingRequest(
        String requestId,
        String model,
        List<String> inputs,
        Map<String, Object> parameters) {

    public EmbeddingRequest {
        if (requestId == null) {
            requestId = UUID.randomUUID().toString();
        }
        Objects.requireNonNull(model, "model is required");
        Objects.requireNonNull(inputs, "inputs is required");
        if (inputs.isEmpty()) {
            throw new IllegalArgumentException("inputs must not be empty");
        }
        if (parameters == null) {
            parameters = Map.of();
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String requestId;
        private String model;
        private List<String> inputs;
        private Map<String, Object> parameters = Map.of();

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder inputs(List<String> inputs) {
            this.inputs = inputs;
            return this;
        }

        public Builder input(String input) {
            this.inputs = List.of(input);
            return this;
        }

        public Builder parameters(Map<String, Object> parameters) {
            this.parameters = parameters;
            return this;
        }

        public EmbeddingRequest build() {
            return new EmbeddingRequest(requestId, model, inputs, parameters);
        }
    }
}
