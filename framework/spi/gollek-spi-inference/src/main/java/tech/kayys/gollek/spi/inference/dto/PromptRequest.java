package tech.kayys.gollek.spi.inference.dto;

import java.util.List;
import java.util.Map;

/**
 * Modern inference request DTO for high-performance runtimes.
 */
public record PromptRequest(
    String model,
    List<ChatMessage> messages,
    int maxTokens,
    double temperature,
    double topP,
    List<String> stop,
    boolean stream,
    Map<String, Object> extra
) {
    public PromptRequest {
        if (maxTokens <= 0) maxTokens = 256;
        if (temperature <= 0) temperature = 1.0;
        if (topP <= 0 || topP > 1.0) topP = 1.0;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String model;
        private List<ChatMessage> messages;
        private int maxTokens = 256;
        private double temperature = 1.0;
        private double topP = 1.0;
        private List<String> stop;
        private boolean stream;
        private Map<String, Object> extra;

        public Builder model(String model) { this.model = model; return this; }
        public Builder messages(List<ChatMessage> messages) { this.messages = messages; return this; }
        public Builder maxTokens(int maxTokens) { this.maxTokens = maxTokens; return this; }
        public Builder temperature(double temperature) { this.temperature = temperature; return this; }
        public Builder topP(double topP) { this.topP = topP; return this; }
        public Builder stop(List<String> stop) { this.stop = stop; return this; }
        public Builder stream(boolean stream) { this.stream = stream; return this; }
        public Builder extra(Map<String, Object> extra) { this.extra = extra; return this; }

        public PromptRequest build() {
            return new PromptRequest(model, messages, maxTokens, temperature, topP, stop, stream, extra);
        }
    }
}
