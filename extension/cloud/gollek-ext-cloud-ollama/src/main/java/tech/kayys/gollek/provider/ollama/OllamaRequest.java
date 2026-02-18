package tech.kayys.gollek.provider.ollama;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * Ollama chat request
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class OllamaRequest {

    private String model;
    private List<OllamaMessage> messages;
    private boolean stream;
    private String format;
    private OllamaOptions options;

    @JsonProperty("keep_alive")
    private String keepAlive;

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public List<OllamaMessage> getMessages() {
        return messages;
    }

    public void setMessages(List<OllamaMessage> messages) {
        this.messages = messages;
    }

    public boolean isStream() {
        return stream;
    }

    public void setStream(boolean stream) {
        this.stream = stream;
    }

    public String getFormat() {
        return format;
    }

    public void setFormat(String format) {
        this.format = format;
    }

    public OllamaOptions getOptions() {
        return options;
    }

    public void setOptions(OllamaOptions options) {
        this.options = options;
    }

    public String getKeepAlive() {
        return keepAlive;
    }

    public void setKeepAlive(String keepAlive) {
        this.keepAlive = keepAlive;
    }
}