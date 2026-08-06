package tech.kayys.gollek.sdk.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder;

import java.util.List;

/**
 * Request for generating vector embeddings from text.
 *
 * <p>Use {@link #of(String)} for a quick single-input request, or {@link #builder()}
 * to specify a model.
 *
 * @param input the text to embed
 * @param model the embedding model identifier; may be {@code null} to use the client default
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonDeserialize(builder = EmbeddingRequest.Builder.class)
public record EmbeddingRequest(
        String input,
        String model
) {

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates an embedding request for the given input using the client's default model.
     *
     * @param input the text to embed
     * @return a minimal {@link EmbeddingRequest}
     */
    public static EmbeddingRequest of(String input) {
        return builder().input(input).build();
    }

    @JsonPOJOBuilder(withPrefix = "")
    public static class Builder {
        private String input;
        private String model;

        public Builder input(String input) {
            this.input = input;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public EmbeddingRequest build() {
            return new EmbeddingRequest(input, model);
        }
    }
}
