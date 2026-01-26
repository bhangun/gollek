package tech.kayys.golek.model;

import java.util.List;

public record EmbeddingResult(
    List<float[]> embeddings,
    int dimensions,
    long timeMs
) {}
