package tech.kayys.gollek.provider.anthropic;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * Anthropic messages response.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class AnthropicResponse {

    @JsonProperty("id")
    private String id;

    @JsonProperty("type")
    private String type;

    @JsonProperty("role")
    private String role;

    @JsonProperty("model")
    private String model;

    @JsonProperty("content")
    private List<AnthropicContentBlock> content;

    @JsonProperty("usage")
    private AnthropicUsage usage;

    @JsonProperty("stop_reason")
    private String stopReason;

    public AnthropicResponse() {
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getRole() {
        return role;
    }

    public void setRole(String role) {
        this.role = role;
    }

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public List<AnthropicContentBlock> getContent() {
        return content;
    }

    public void setContent(List<AnthropicContentBlock> content) {
        this.content = content;
    }

    public AnthropicUsage getUsage() {
        return usage;
    }

    public void setUsage(AnthropicUsage usage) {
        this.usage = usage;
    }

    public String getStopReason() {
        return stopReason;
    }

    public void setStopReason(String stopReason) {
        this.stopReason = stopReason;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class AnthropicContentBlock {
        @JsonProperty("type")
        private String type;

        @JsonProperty("text")
        private String text;

        public String getType() {
            return type;
        }

        public void setType(String type) {
            this.type = type;
        }

        public String getText() {
            return text;
        }

        public void setText(String text) {
            this.text = text;
        }
    }
}
