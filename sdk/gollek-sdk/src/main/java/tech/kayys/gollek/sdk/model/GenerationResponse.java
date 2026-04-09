package tech.kayys.gollek.sdk.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder;

/**
 * Encapsulates the model's generated text and token usage metrics.
 *
 * @param text  the generated text content
 * @param model the model identifier that produced the response
 * @param usage token usage statistics for the request/response pair
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonDeserialize(builder = GenerationResponse.Builder.class)
public record GenerationResponse(
        String text,
        String model,
        Usage usage
) {

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Token usage breakdown for a single inference call.
     *
     * @param promptTokens     number of tokens in the input prompt
     * @param completionTokens number of tokens in the generated output
     * @param totalTokens      sum of prompt and completion tokens
     */
    public record Usage(
            int promptTokens,
            int completionTokens,
            int totalTokens
    ) {
        public static Builder builder() {
            return new Builder();
        }

        @JsonPOJOBuilder(withPrefix = "")
        public static class Builder {
            private int promptTokens;
            private int completionTokens;
            private int totalTokens;

            public Builder promptTokens(int promptTokens) {
                this.promptTokens = promptTokens;
                return this;
            }

            public Builder completionTokens(int completionTokens) {
                this.completionTokens = completionTokens;
                return this;
            }

            public Builder totalTokens(int totalTokens) {
                this.totalTokens = totalTokens;
                return this;
            }

            public Usage build() {
                return new Usage(promptTokens, completionTokens, totalTokens);
            }
        }
    }

    @JsonPOJOBuilder(withPrefix = "")
    public static class Builder {
        private String text;
        private String model;
        private Usage usage;

        public Builder text(String text) {
            this.text = text;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder usage(Usage usage) {
            this.usage = usage;
            return this;
        }

        public GenerationResponse build() {
            return new GenerationResponse(text, model, usage);
        }
    }
}
