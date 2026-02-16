package tech.kayys.golek.provider.ollama;

import com.fasterxml.jackson.annotation.JsonProperty;

public class OllamaPullRequest {
    @JsonProperty("name")
    private String name;

    @JsonProperty("stream")
    private boolean stream = true;

    @JsonProperty("insecure")
    private boolean insecure;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public boolean isStream() {
        return stream;
    }

    public void setStream(boolean stream) {
        this.stream = stream;
    }

    public boolean isInsecure() {
        return insecure;
    }

    public void setInsecure(boolean insecure) {
        this.insecure = insecure;
    }
}
