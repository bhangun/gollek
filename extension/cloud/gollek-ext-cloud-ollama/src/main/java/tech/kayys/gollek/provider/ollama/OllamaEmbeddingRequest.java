package tech.kayys.gollek.provider.ollama;

import com.fasterxml.jackson.annotation.JsonProperty;

public class OllamaEmbeddingRequest {
    @JsonProperty("model")
    private String model;

    @JsonProperty("prompt")
    private String prompt;

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public String getPrompt() {
        return prompt;
    }

    public void setPrompt(String prompt) {
        this.prompt = prompt;
    }
}
