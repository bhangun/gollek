package tech.kayys.gollek.provider.gemini;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public class GeminiModelInfo {
    @JsonProperty("name")
    private String name;

    @JsonProperty("version")
    private String version;

    @JsonProperty("displayName")
    private String displayName;

    @JsonProperty("description")
    private String description;

    @JsonProperty("inputTokenLimit")
    private Integer inputTokenLimit;

    @JsonProperty("outputTokenLimit")
    private Integer outputTokenLimit;

    @JsonProperty("supportedGenerationMethods")
    private List<String> supportedGenerationMethods;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getVersion() {
        return version;
    }

    public void setVersion(String version) {
        this.version = version;
    }

    public String getDisplayName() {
        return displayName;
    }

    public void setDisplayName(String displayName) {
        this.displayName = displayName;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public Integer getInputTokenLimit() {
        return inputTokenLimit;
    }

    public void setInputTokenLimit(Integer inputTokenLimit) {
        this.inputTokenLimit = inputTokenLimit;
    }

    public Integer getOutputTokenLimit() {
        return outputTokenLimit;
    }

    public void setOutputTokenLimit(Integer outputTokenLimit) {
        this.outputTokenLimit = outputTokenLimit;
    }

    public List<String> getSupportedGenerationMethods() {
        return supportedGenerationMethods;
    }

    public void setSupportedGenerationMethods(List<String> supportedGenerationMethods) {
        this.supportedGenerationMethods = supportedGenerationMethods;
    }
}
