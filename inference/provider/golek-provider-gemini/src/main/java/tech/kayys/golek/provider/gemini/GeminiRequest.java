package tech.kayys.golek.provider.gemini;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.List;

/**
 * Gemini API request
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class GeminiRequest {

    private List<GeminiContent> contents;
    private GeminiGenerationConfig generationConfig;
    private GeminiContent systemInstruction;

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

    public GeminiContent getSystemInstruction() {
        return systemInstruction;
    }

    public void setSystemInstruction(GeminiContent systemInstruction) {
        this.systemInstruction = systemInstruction;
    }
}
