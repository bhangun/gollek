package tech.kayys.gollek.provider.gemini;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public class GeminiModelsResponse {
    @JsonProperty("models")
    private List<GeminiModelInfo> models;

    public List<GeminiModelInfo> getModels() {
        return models;
    }

    public void setModels(List<GeminiModelInfo> models) {
        this.models = models;
    }
}
