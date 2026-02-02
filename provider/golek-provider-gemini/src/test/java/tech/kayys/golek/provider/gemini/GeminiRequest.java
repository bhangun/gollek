package tech.kayys.golek.provider.gemini;

import java.util.List;

public class GeminiRequest {
    private List<GeminiContent> contents;
    private GeminiGenerationConfig generationConfig;

    public List<GeminiContent> getContents() {
        return contents;
    }

    public void setContents(List<GeminiContent> contents) {
        this.contents = contents;
    }

    public GeminiGenerationConfig getGenerationConfig() {
        return generationConfig;
    }

    public void setGenerationConfig(GeminiGenerationConfig generationConfig) {
        this.generationConfig = generationConfig;
    }
}
