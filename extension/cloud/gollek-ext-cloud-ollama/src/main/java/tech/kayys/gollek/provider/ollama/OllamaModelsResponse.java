package tech.kayys.gollek.provider.ollama;

import java.util.List;

/**
 * Ollama models list response
 */
public class OllamaModelsResponse {

    private List<OllamaModelInfo> models;

    public List<OllamaModelInfo> getModels() {
        return models;
    }

    public void setModels(List<OllamaModelInfo> models) {
        this.models = models;
    }
}
