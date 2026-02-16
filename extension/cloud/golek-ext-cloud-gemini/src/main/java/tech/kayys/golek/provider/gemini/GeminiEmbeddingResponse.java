package tech.kayys.golek.provider.gemini;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public class GeminiEmbeddingResponse {
    @JsonProperty("embedding")
    private GeminiEmbeddingValue embedding;

    public GeminiEmbeddingValue getEmbedding() {
        return embedding;
    }

    public void setEmbedding(GeminiEmbeddingValue embedding) {
        this.embedding = embedding;
    }

    public static class GeminiEmbeddingValue {
        @JsonProperty("values")
        private List<Double> values;

        public List<Double> getValues() {
            return values;
        }

        public void setValues(List<Double> values) {
            this.values = values;
        }
    }
}
