package tech.kayys.gollek.provider.ollama;

import com.fasterxml.jackson.annotation.JsonProperty;

public class OllamaShowRequest {
    @JsonProperty("name")
    private String name;

    public OllamaShowRequest() {
    }

    public OllamaShowRequest(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
