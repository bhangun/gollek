package tech.kayys.golek.provider.anthropic;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public record AnthropicRequest(
        String model,
        @JsonProperty("messages") List<Message> messages,
        @JsonProperty("max_tokens") int maxTokens,
        @JsonProperty("temperature") double temperature,
        @JsonProperty("stream") boolean stream,
        @JsonProperty("system") String systemPrompt) {

    public AnthropicRequest {
        // Ensure systemPrompt is not null
        if (systemPrompt == null) {
            systemPrompt = "";
        }
    }

    public record Message(@JsonProperty("role") String role, @JsonProperty("content") String content) {
    }

    // Builder pattern for more flexible construction
    public static class Builder {
        private String model;
        private List<Message> messages;
        private int maxTokens = 2048;
        private double temperature = 0.7;
        private boolean stream = false;
        private String systemPrompt = "";

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder messages(List<Message> messages) {
            this.messages = messages;
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

        public Builder stream(boolean stream) {
            this.stream = stream;
            return this;
        }

        public Builder systemPrompt(String systemPrompt) {
            this.systemPrompt = systemPrompt;
            return this;
        }

        public AnthropicRequest build() {
            return new AnthropicRequest(model, messages, maxTokens, temperature, stream, systemPrompt);
        }
    }
}
