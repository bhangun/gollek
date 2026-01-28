package tech.kayys.golek.provider.gemini;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public class GeminiEmbeddingRequest {
    @JsonProperty("content")
    private GeminiContent content;

    public GeminiEmbeddingRequest() {
    }

    public GeminiEmbeddingRequest(GeminiContent content) {
        this.content = content;
    }

    public GeminiContent getContent() {
        return content;
    }

    public void setContent(GeminiContent content) {
        this.content = content;
    }
}
