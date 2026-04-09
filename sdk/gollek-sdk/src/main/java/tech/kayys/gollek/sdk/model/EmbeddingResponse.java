package tech.kayys.gollek.sdk.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder;

/**
 * Result of an embedding generation request.
 *
 * @param vector the dense float vector representing the input text
 * @param model  the embedding model that produced the vector
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonDeserialize(builder = EmbeddingResponse.Builder.class)
public record EmbeddingResponse(
        float[] vector,
        String model
) {

    public static Builder builder() {
        return new Builder();
    }

    @JsonPOJOBuilder(withPrefix = "")
    public static class Builder {
        private float[] vector;
        private String model;

        public Builder vector(float[] vector) {
            this.vector = vector;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public EmbeddingResponse build() {
            return new EmbeddingResponse(vector, model);
        }
    }
}
