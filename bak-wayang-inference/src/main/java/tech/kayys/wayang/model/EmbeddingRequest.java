package tech.kayys.golek.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public record EmbeddingRequest(
    Object input, // String or List<String>
    String model,
    @JsonProperty("encoding_format") String encodingFormat
) {}
