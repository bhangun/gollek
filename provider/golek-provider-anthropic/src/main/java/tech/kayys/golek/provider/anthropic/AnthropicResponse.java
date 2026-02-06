package tech.kayys.golek.provider.anthropic;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public record AnthropicResponse(
        String id,
        String type,
        String role,
        List<Content> content,
        String model,
        @JsonProperty("stop_reason") String stopReason,
        @JsonProperty("stop_sequence") String stopSequence,
        Usage usage) {
    public record Content(String type, String text) {
    }

    public record Usage(
            @JsonProperty("input_tokens") int inputTokens,
            @JsonProperty("output_tokens") int outputTokens) {
    }
}
