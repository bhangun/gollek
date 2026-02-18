package tech.kayys.gollek.provider.ollama;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * Ollama model details
 */
public class OllamaModelDetails {

    @JsonProperty("parent_model")
    private String parentModel;

    private String format;
    private String family;

    @JsonProperty("parameter_size")
    private String parameterSize;

    @JsonProperty("quantization_level")
    private String quantizationLevel;

    private List<String> families;

    public String getParentModel() {
        return parentModel;
    }

    public void setParentModel(String parentModel) {
        this.parentModel = parentModel;
    }

    public String getFormat() {
        return format;
    }

    public void setFormat(String format) {
        this.format = format;
    }

    public String getFamily() {
        return family;
    }

    public void setFamily(String family) {
        this.family = family;
    }

    public String getParameterSize() {
        return parameterSize;
    }

    public void setParameterSize(String parameterSize) {
        this.parameterSize = parameterSize;
    }

    public String getQuantizationLevel() {
        return quantizationLevel;
    }

    public void setQuantizationLevel(String quantizationLevel) {
        this.quantizationLevel = quantizationLevel;
    }

    public List<String> getFamilies() {
        return families;
    }

    public void setFamilies(List<String> families) {
        this.families = families;
    }
}
