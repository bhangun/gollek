package tech.kayys.golek.model;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;

public record EmbeddingResponse(
    String object,
    List<Embedding> data,
    String model,
    Usage usage
) {
    public record Embedding(
        String object,
        int index,
        float[] embedding
    ) {}
    
    public record Usage(
        @JsonProperty("prompt_tokens") int promptTokens,
        @JsonProperty("total_tokens") int totalTokens
    ) {}
}