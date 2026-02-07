package tech.kayys.golek.provider.mistral;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import java.util.Map;

public class MistralRequest {
    private String model;
    private List<MistralMessage> messages;
    private Double temperature;
    @JsonProperty("max_tokens")
    private Integer maxTokens;
    private boolean stream;

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public List<MistralMessage> getMessages() {
        return messages;
    }

    public void setMessages(List<MistralMessage> messages) {
        this.messages = messages;
    }

    public Double getTemperature() {
        return temperature;
    }

    public void setTemperature(Double temperature) {
        this.temperature = temperature;
    }

    public Integer getMaxTokens() {
        return maxTokens;
    }

    public void setMaxTokens(Integer maxTokens) {
        this.maxTokens = maxTokens;
    }

    public boolean isStream() {
        return stream;
    }

    public void setStream(boolean stream) {
        this.stream = stream;
    }
}
