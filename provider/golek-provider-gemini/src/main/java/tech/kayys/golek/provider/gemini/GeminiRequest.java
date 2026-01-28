package tech.kayys.golek.provider.gemini;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.List;

/**
 * Gemini API request
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class GeminiRequest {

    private List<GeminiContent> contents;
    private List<GeminiTool> tools;
    private GeminiGenerationConfig generationConfig;
    private List<GeminiSafetySettings> safetySettings;
    private GeminiContent systemInstruction;

    public List<GeminiContent> getContents() {
        return contents;
    }

    public void setContents(List<GeminiContent> contents) {
        this.contents = contents;
    }

    public List<GeminiTool> getTools() {
        return tools;
    }

    public void setTools(List<GeminiTool> tools) {
        this.tools = tools;
    }

    public GeminiGenerationConfig getGenerationConfig() {
        return generationConfig;
    }

    public void setGenerationConfig(GeminiGenerationConfig generationConfig) {
        this.generationConfig = generationConfig;
    }

    public List<GeminiSafetySettings> getSafetySettings() {
        return safetySettings;
    }

    public void setSafetySettings(List<GeminiSafetySettings> safetySettings) {
        this.safetySettings = safetySettings;
    }

    public GeminiContent getSystemInstruction() {
        return systemInstruction;
    }

    public void setSystemInstruction(GeminiContent systemInstruction) {
        this.systemInstruction = systemInstruction;
    }
}
