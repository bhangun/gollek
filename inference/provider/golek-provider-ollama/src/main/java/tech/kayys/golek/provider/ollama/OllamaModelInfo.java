package tech.kayys.golek.provider.ollama;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Ollama model info
 */
public class OllamaModelInfo {

    private String name;
    private String model;

    @JsonProperty("modified_at")
    private String modifiedAt;

    private long size;
    private String digest;
    private OllamaModelDetails details;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public String getModifiedAt() {
        return modifiedAt;
    }

    public void setModifiedAt(String modifiedAt) {
        this.modifiedAt = modifiedAt;
    }

    public long getSize() {
        return size;
    }

    public void setSize(long size) {
        this.size = size;
    }

    public String getDigest() {
        return digest;
    }

    public void setDigest(String digest) {
        this.digest = digest;
    }

    public OllamaModelDetails getDetails() {
        return details;
    }

    public void setDetails(OllamaModelDetails details) {
        this.details = details;
    }
}
