package tech.kayys.golek.provider.ollama;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public class OllamaEmbeddingResponse {
    @JsonProperty("embedding")
    private List<Double> embedding;

    public List<Double> getEmbedding() {
        return embedding;
    }

    public void setEmbedding(List<Double> embedding) {
        this.embedding = embedding;
    }
}
